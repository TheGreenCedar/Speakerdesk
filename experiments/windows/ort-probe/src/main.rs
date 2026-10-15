//! Native graph qualification; model-specific exporters own numerical acceptance.
use ort::{AsPointer, session::Session, value::Tensor};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs,
    io::{Read, Write},
    path::Path,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    model: String,
    #[serde(default)]
    model_files: std::collections::BTreeMap<String, String>,
    inputs: Vec<Input>,
    repeat: usize,
    #[serde(default)]
    disable_optimizations: bool,
    #[serde(default)]
    require_dml_only: bool,
    #[serde(default)]
    disable_memory_arena: bool,
    #[serde(default)]
    disable_graph_fusion: bool,
    #[serde(default)]
    idle_ms: u64,
    #[serde(default)]
    inspect_placement_only: bool,
    #[serde(default)]
    placement_policy: Option<PlacementPolicy>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
struct NodeIdentity {
    name: String,
    domain: String,
    op: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PolicyNode {
    node: NodeIdentity,
    cpu_shape: bool,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PlacementPolicy {
    model_sha256: String,
    // Exact surviving assignment snapshot, reviewed against the pinned graph;
    // omissions for required folding/deletion are never inferred by the probe.
    nodes: Vec<PolicyNode>,
}

#[derive(Clone, Debug, Serialize)]
struct Assignment {
    node: NodeIdentity,
    provider: String,
    subgraph: usize,
}

fn validate_assignment_table(assignments: &[Assignment]) -> Result<()> {
    let mut names = std::collections::BTreeSet::new();
    if assignments.is_empty() {
        return Err("runtime returned no placement assignments".into());
    }
    for item in assignments {
        if item.node.name.is_empty() || item.node.op.is_empty() || !names.insert(&item.node.name) {
            return Err("placement contains an unnamed, operatorless or duplicate node".into());
        }
        if !matches!(
            item.provider.as_str(),
            "DmlExecutionProvider" | "CPUExecutionProvider"
        ) {
            return Err(format!("unrecognized assigned provider: {}", item.provider).into());
        }
    }
    Ok(())
}

fn admit_placement(
    policy: &PlacementPolicy,
    model_sha256: &str,
    assignments: &[Assignment],
) -> Result<()> {
    validate_assignment_table(assignments)?;
    if policy.model_sha256 != model_sha256 {
        return Err("placement policy belongs to a different model hash".into());
    }
    let mut expected = std::collections::BTreeMap::new();
    let mut names = std::collections::BTreeSet::new();
    for item in &policy.nodes {
        if item.node.name.is_empty()
            || item.node.op.is_empty()
            || !names.insert(&item.node.name)
            || expected.insert(item.node.clone(), item.cpu_shape).is_some()
        {
            return Err(
                "placement policy contains an unnamed, operatorless or duplicate node".into(),
            );
        }
        // The graph-specific cpu_shape assertion also requires a narrow op class.
        // Integer dtype and shape ancestry are reviewed separately against its hash.
        if item.cpu_shape
            && (item.node.domain != ""
                || !matches!(
                    item.node.op.as_str(),
                    "Shape" | "Size" | "Gather" | "Unsqueeze" | "Concat" | "Cast" | "Sub" | "Neg"
                ))
        {
            return Err("CPU shape admission names an unreviewed operator class".into());
        }
    }
    if expected.len() != assignments.len() {
        return Err("placement snapshot has missing or extra assignments".into());
    }
    let mut accelerated_numerical = false;
    for item in assignments {
        let cpu_shape = expected
            .remove(&item.node)
            .ok_or("unexpected assigned node identity")?;
        if item.provider == "CPUExecutionProvider" && !cpu_shape {
            return Err(format!("numerical node assigned to CPU: {}", item.node.name).into());
        }
        accelerated_numerical |= item.provider == "DmlExecutionProvider" && !cpu_shape;
    }
    if !expected.is_empty() || !accelerated_numerical {
        return Err("placement is incomplete or has no accelerated numerical work".into());
    }
    Ok(())
}

fn placement_assignments(session: &Session) -> Result<Vec<Assignment>> {
    use std::{ffi::CStr, ptr, slice};
    let api = ort::api();
    let mut groups: *const *const ort::sys::OrtEpAssignedSubgraph = ptr::null();
    let mut count = 0;
    let mut assignments = Vec::new();
    // ORT owns these arrays/strings for the live session; copy them before drop.
    unsafe {
        ort::Error::result_from_status((api.Session_GetEpGraphAssignmentInfo)(
            session.ptr(),
            &mut groups,
            &mut count,
        ))?;
        if groups.is_null() || count == 0 || count > 1_000_000 {
            return Err("runtime returned no valid assignment subgraphs".into());
        }
        for (subgraph, &group) in slice::from_raw_parts(groups, count).iter().enumerate() {
            if group.is_null() {
                return Err("null assigned subgraph".into());
            }
            let mut provider = ptr::null();
            ort::Error::result_from_status((api.EpAssignedSubgraph_GetEpName)(
                group,
                &mut provider,
            ))?;
            if provider.is_null() {
                return Err("null assigned provider name".into());
            }
            let provider = CStr::from_ptr(provider).to_str()?.to_owned();
            let mut nodes: *const *const ort::sys::OrtEpAssignedNode = ptr::null();
            let mut node_count = 0;
            ort::Error::result_from_status((api.EpAssignedSubgraph_GetNodes)(
                group,
                &mut nodes,
                &mut node_count,
            ))?;
            if nodes.is_null() || node_count == 0 || node_count > 1_000_000 {
                return Err("runtime returned an empty or invalid assigned subgraph".into());
            }
            for &node in slice::from_raw_parts(nodes, node_count) {
                if node.is_null() {
                    return Err("null assigned node".into());
                }
                let (mut name, mut domain, mut op) = (ptr::null(), ptr::null(), ptr::null());
                ort::Error::result_from_status((api.EpAssignedNode_GetName)(node, &mut name))?;
                ort::Error::result_from_status((api.EpAssignedNode_GetDomain)(node, &mut domain))?;
                ort::Error::result_from_status((api.EpAssignedNode_GetOperatorType)(
                    node, &mut op,
                ))?;
                if name.is_null() || domain.is_null() || op.is_null() {
                    return Err("null assigned node identity".into());
                }
                assignments.push(Assignment {
                    node: NodeIdentity {
                        name: CStr::from_ptr(name).to_str()?.to_owned(),
                        domain: CStr::from_ptr(domain).to_str()?.to_owned(),
                        op: CStr::from_ptr(op).to_str()?.to_owned(),
                    },
                    provider: provider.clone(),
                    subgraph,
                });
            }
        }
    }
    Ok(assignments)
}

fn idle_phase(output: &Path, name: &str, milliseconds: u64) -> Result<()> {
    if milliseconds != 0 {
        let phase = json!({"phase": name, "pid": std::process::id(),
            "unix_ms": SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
            "idle_ms": milliseconds});
        fs::write(
            output.join(format!("phase-{name}.json")),
            serde_json::to_vec_pretty(&phase)?,
        )?;
        println!("{}", phase);
        std::io::stdout().flush()?;
        std::thread::sleep(Duration::from_millis(milliseconds));
    }
    Ok(())
}

fn private_commit() -> Result<serde_json::Value> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::{
            ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX},
            Threading::GetCurrentProcess,
        };
        let mut counters: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
        let bytes = std::mem::size_of_val(&counters) as u32;
        counters.cb = bytes;
        if unsafe {
            GetProcessMemoryInfo(
                GetCurrentProcess(),
                (&mut counters as *mut PROCESS_MEMORY_COUNTERS_EX).cast(),
                bytes,
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
        // PeakPagefileUsage is lifetime peak private commit, not resident RAM or GPU peak.
        Ok(json!({"current_bytes": counters.PrivateUsage,
            "lifetime_peak_bytes": counters.PeakPagefileUsage,
            "source": "GetProcessMemoryInfo/PROCESS_MEMORY_COUNTERS_EX"}))
    }
    #[cfg(not(windows))]
    Ok(serde_json::Value::Null)
}

fn run_phase(output: &Path, run: usize, completed: bool) -> Result<()> {
    let name = if completed { "complete" } else { "start" };
    let phase = json!({"phase": format!("run-{name}"), "run": run,
        "pid": std::process::id(), "unix_ms": SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
        "model_run_returned": completed, "private_commit": private_commit()?});
    fs::write(
        output.join(format!("phase-run-{run}-{name}.json")),
        serde_json::to_vec_pretty(&phase)?,
    )?;
    println!("{phase}");
    std::io::stdout().flush()?;
    Ok(())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    name: String,
    dtype: String,
    shape: Vec<i64>,
    path: String,
}

fn sha256(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            return Ok(format!("{:x}", hash.finalize()));
        }
        hash.update(&buffer[..count]);
    }
}

fn check_tensor(shape: &[i64], bytes: usize, width: usize) -> Result<()> {
    let elements = shape.iter().try_fold(1_usize, |total, &dim| {
        usize::try_from(dim)
            .ok()
            .and_then(|dim| total.checked_mul(dim))
    });
    if elements.and_then(|count| count.checked_mul(width)) != Some(bytes) {
        return Err("tensor shape does not match its file length".into());
    }
    Ok(())
}

#[cfg(windows)]
fn secure_dll_search(runtime: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::System::LibraryLoader::{
        AddDllDirectory, LOAD_LIBRARY_SEARCH_SYSTEM32, LOAD_LIBRARY_SEARCH_USER_DIRS,
        SetDefaultDllDirectories,
    };
    let directory = runtime.parent().ok_or("runtime has no parent")?;
    let wide: Vec<u16> = directory.as_os_str().encode_wide().chain(Some(0)).collect();
    // Exclude the working directory and PATH from transitive runtime DLL lookup.
    unsafe {
        if SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32 | LOAD_LIBRARY_SEARCH_USER_DIRS)
            == 0
            || AddDllDirectory(wide.as_ptr()).is_null()
        {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    Ok(())
}

#[cfg(windows)]
fn loaded_runtime_libraries() -> Result<serde_json::Value> {
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    use windows_sys::Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleW};
    let mut libraries = Vec::new();
    for name in [
        "onnxruntime.dll",
        "DirectML.dll",
        "Microsoft.Windows.AI.MachineLearning.dll",
    ] {
        let wide: Vec<u16> = std::ffi::OsStr::new(name)
            .encode_wide()
            .chain(Some(0))
            .collect();
        let module = unsafe { GetModuleHandleW(wide.as_ptr()) };
        if module.is_null() {
            continue;
        }
        let mut buffer = vec![0_u16; 32768];
        let length = unsafe { GetModuleFileNameW(module, buffer.as_mut_ptr(), buffer.len() as u32) }
            as usize;
        if length == 0 || length == buffer.len() {
            return Err(std::io::Error::last_os_error().into());
        }
        let path = std::path::PathBuf::from(std::ffi::OsString::from_wide(&buffer[..length]));
        libraries.push(json!({"name": name, "path": path, "sha256": sha256(&path)?}));
    }
    Ok(json!(libraries))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    if !(5..=6).contains(&args.len()) {
        return Err(
            "usage: ort-probe <runtime-dll> <cpu|dml> <case.json> <output-directory> [adapter]"
                .into(),
        );
    }
    let runtime = Path::new(&args[1]).canonicalize()?;
    #[cfg(windows)]
    secure_dll_search(&runtime)?;
    let provider = args[2].to_str().ok_or("invalid provider")?;
    if !matches!(provider, "cpu" | "dml") {
        return Err("provider must be cpu or dml".into());
    }
    let case_path = Path::new(&args[3]).canonicalize()?;
    let root = case_path.parent().ok_or("case has no parent")?;
    let case: Case = serde_json::from_slice(&fs::read(&case_path)?)?;
    if !(1..=1000).contains(&case.repeat) || case.inputs.is_empty() {
        return Err("case requires inputs and 1..=1000 repetitions".into());
    }
    if case.idle_ms > 5000 || (case.require_dml_only && provider != "dml") {
        return Err("idle must be <=5000ms and strict DirectML requires dml".into());
    }
    let capture_placement = case.inspect_placement_only || case.placement_policy.is_some();
    if capture_placement
        && (provider != "dml"
            || case.require_dml_only
            || !case.disable_optimizations
            || !case.disable_memory_arena)
    {
        return Err("placement inspection/admission requires dml, disabled optimization/arena, and no conflicting global strict flag".into());
    }
    let output = Path::new(&args[4]);
    fs::create_dir(output)?;
    let output = output.canonicalize()?;
    idle_phase(&output, "baseline", case.idle_ms)?;
    let environment = ort::init_from(&runtime)?.with_telemetry(false).build()?;
    let mut builder = Session::builder(&environment)?
        .with_parallel_execution(false)?
        .with_memory_pattern(false)?
        .with_intra_threads(4)?
        .with_profiling(output.join("profile"))?;
    if case.disable_optimizations {
        builder = builder
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Disable)?;
    }
    if case.require_dml_only {
        builder = builder.with_config_entry("session.disable_cpu_ep_fallback", "1")?;
    }
    if case.disable_memory_arena {
        builder = builder.with_config_entry("ep.dml.disable_memory_arena", "1")?;
    }
    if case.disable_graph_fusion {
        builder = builder.with_config_entry("ep.dml.disable_graph_fusion", "1")?;
    }
    if capture_placement {
        builder = builder.with_config_entry("session.record_ep_graph_assignment_info", "1")?;
    }
    if provider == "dml" {
        let adapter: i32 = args
            .get(5)
            .map(|value| value.to_string_lossy().parse())
            .transpose()?
            .unwrap_or(0);
        builder = builder.with_execution_providers([ort::ep::DirectML::default()
            .with_device_id(adapter)
            .build()
            .error_on_failure()])?;
    }
    let model = root.join(&case.model).canonicalize()?;
    let model_sha256 = sha256(&model)?;
    if case
        .placement_policy
        .as_ref()
        .is_some_and(|policy| policy.model_sha256 != model_sha256)
    {
        return Err("placement policy belongs to a different model hash".into());
    }
    let mut model_files = std::collections::BTreeMap::new();
    model_files.insert(case.model.clone(), sha256(&model)?);
    for (name, expected) in &case.model_files {
        let actual = sha256(&root.join(name))?;
        if &actual != expected {
            return Err(format!("model artifact hash differs: {name}").into());
        }
        model_files.insert(name.clone(), actual);
    }
    let started = Instant::now();
    let mut session = builder.commit_from_file(&model)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.;
    let assignments = if capture_placement {
        let assignments = placement_assignments(&session)?;
        #[cfg(windows)]
        let libraries = loaded_runtime_libraries()?;
        #[cfg(not(windows))]
        let libraries = serde_json::Value::Null;
        let admission = validate_assignment_table(&assignments).and_then(|()| {
            if let Some(policy) = &case.placement_policy {
                admit_placement(policy, &model_sha256, &assignments)
            } else {
                Ok(())
            }
        });
        fs::write(
            output.join("placement.json"),
            serde_json::to_vec_pretty(&json!({
                "pid": std::process::id(), "model_sha256": model_sha256,
                "runtime_sha256": sha256(&runtime)?, "loaded_runtime_libraries": libraries,
                "runner_sha256": sha256(&std::env::current_exe()?)?, "case_sha256": sha256(&case_path)?,
                "model_files": model_files, "assignments": assignments, "load_ms": load_ms,
                "inspection_only": case.inspect_placement_only,
                "policy_admitted": case.placement_policy.is_some() && admission.is_ok(),
                "admission_error": admission.as_ref().err().map(|error| error.to_string()),
                "snapshot": "ORT required/minimal optimization snapshot; compiled groups expose their original constituent node identities",
                "model_run_called": false
            }))?,
        )?;
        admission?;
        if case.inspect_placement_only {
            println!("Placement inspection saved; model Run was not called.");
            return Ok(());
        }
        Some(assignments)
    } else {
        None
    };
    idle_phase(&output, "loaded", case.idle_ms)?;
    let mut inputs = Vec::new();
    let mut input_records = Vec::new();
    for input in &case.inputs {
        let path = root.join(&input.path);
        let bytes = fs::read(&path)?;
        let value = match input.dtype.as_str() {
            "f32" => {
                check_tensor(&input.shape, bytes.len(), 4)?;
                let values: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(format!("nonfinite input {}", input.name).into());
                }
                Tensor::from_array((input.shape.clone(), values))?.into_dyn()
            }
            "i64" => {
                check_tensor(&input.shape, bytes.len(), 8)?;
                let values: Vec<i64> = bytes
                    .chunks_exact(8)
                    .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                Tensor::from_array((input.shape.clone(), values))?.into_dyn()
            }
            _ => return Err(format!("unsupported input type {}", input.dtype).into()),
        };
        input_records.push(json!({"name": input.name, "dtype": input.dtype, "shape": input.shape, "sha256": sha256(&path)?}));
        inputs.push((input.name.clone(), value));
    }
    let mut run_ms = Vec::new();
    let mut output_records = Vec::new();
    for run in 0..case.repeat {
        run_phase(&output, run, false)?;
        let started = Instant::now();
        let outputs = session.run(
            inputs
                .iter()
                .map(|(name, value)| (name.as_str(), value))
                .collect::<Vec<_>>(),
        )?;
        run_ms.push(started.elapsed().as_secs_f64() * 1000.);
        run_phase(&output, run, true)?;
        for (index, (name, value)) in outputs.iter().enumerate() {
            let (shape, dtype, bytes) = if let Ok((shape, data)) = value.try_extract_tensor::<f32>()
            {
                if data.iter().any(|v| !v.is_finite()) {
                    return Err(format!("nonfinite output {name}").into());
                }
                (
                    shape.to_vec(),
                    "f32",
                    data.iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
            } else {
                let (shape, data) = value.try_extract_tensor::<i64>()?;
                (
                    shape.to_vec(),
                    "i64",
                    data.iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
            };
            let file = format!("run-{run}-output-{index}.{dtype}");
            fs::write(output.join(&file), bytes)?;
            output_records.push(json!({"run": run, "name": name, "shape": shape, "dtype": dtype, "file": file, "sha256": sha256(&output.join(&file))?}));
        }
    }
    idle_phase(&output, "warm", case.idle_ms)?;
    let profile = session.end_profiling()?;
    let events: Vec<serde_json::Value> = serde_json::from_slice(&fs::read(&profile)?)?;
    let mut placement = std::collections::BTreeMap::<String, usize>::new();
    for event in events {
        if let Some(provider) = event["args"]["provider"].as_str() {
            *placement.entry(provider.to_owned()).or_default() += 1;
        }
    }
    #[cfg(windows)]
    let libraries = loaded_runtime_libraries()?;
    #[cfg(not(windows))]
    let libraries = serde_json::Value::Null;
    let placed = (provider != "dml"
        || placement.get("DmlExecutionProvider").copied().unwrap_or(0) != 0)
        && (!case.require_dml_only || placement.keys().all(|key| key == "DmlExecutionProvider"))
        && (case.placement_policy.is_none()
            || placement.keys().all(|key| {
                matches!(
                    key.as_str(),
                    "DmlExecutionProvider" | "CPUExecutionProvider"
                )
            }));
    let report = json!({
        "pid": std::process::id(),
        "runtime": runtime, "runtime_sha256": sha256(&runtime)?, "runtime_info": format!("{:?}", ort::info()),
        "provider_requested": provider, "model": model, "model_sha256": sha256(&model)?,
        "model_files": model_files, "loaded_runtime_libraries": libraries,
        "runner_sha256": sha256(&std::env::current_exe()?)?, "execution_verified": placed,
        "case_sha256": sha256(&case_path)?, "inputs": input_records, "outputs": output_records,
        "graph_optimizations_disabled": case.disable_optimizations,
        "require_dml_only": case.require_dml_only, "memory_arena_disabled": case.disable_memory_arena,
        "dml_graph_fusion_disabled": case.disable_graph_fusion,
        "private_commit": private_commit()?,
        "idle_ms": case.idle_ms,
        "placement_policy_admitted": case.placement_policy.is_some(), "pre_run_assignments": assignments,
        "load_ms": load_ms, "run_ms": run_ms, "executed_provider_events": placement, "profile": profile
    });
    fs::write(
        output.join(if placed { "result.json" } else { "failed.json" }),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !placed {
        return Err(
            "DirectML execution policy failed; inspect the recorded provider events".into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Assignment, NodeIdentity, PlacementPolicy, PolicyNode, admit_placement};

    fn reviewed_placement() -> (PlacementPolicy, Vec<Assignment>) {
        let nodes = vec![
            PolicyNode {
                node: NodeIdentity {
                    name: "shape/gather".into(),
                    domain: "".into(),
                    op: "Gather".into(),
                },
                cpu_shape: true,
            },
            PolicyNode {
                node: NodeIdentity {
                    name: "weights/dequantize".into(),
                    domain: "".into(),
                    op: "DequantizeLinear".into(),
                },
                cpu_shape: false,
            },
            PolicyNode {
                node: NodeIdentity {
                    name: "projection".into(),
                    domain: "".into(),
                    op: "MatMul".into(),
                },
                cpu_shape: false,
            },
        ];
        let assignments = nodes
            .iter()
            .enumerate()
            .map(|(index, node)| Assignment {
                node: node.node.clone(),
                provider: if index == 0 {
                    "CPUExecutionProvider".into()
                } else {
                    "DmlExecutionProvider".into()
                },
                subgraph: usize::from(index != 0),
            })
            .collect();
        (
            PlacementPolicy {
                model_sha256: "reviewed-graph-sha256".into(),
                nodes,
            },
            assignments,
        )
    }

    #[test]
    fn accepts_reviewed_shape_cpu_and_accelerated_reconstruction_before_run() {
        let (policy, mut assignments) = reviewed_placement();
        // Provider group ordering is not part of graph identity; constituent names are.
        assignments.reverse();
        admit_placement(&policy, "reviewed-graph-sha256", &assignments).unwrap();
        assignments[2].provider = "DmlExecutionProvider".into();
        admit_placement(&policy, "reviewed-graph-sha256", &assignments).unwrap();
    }

    #[test]
    fn rejects_hostile_or_incomplete_placement_admission_matrix() {
        for (attack, expected_error) in [
            ("different graph", "different model hash"),
            ("no assignments", "no placement assignments"),
            ("duplicate assignment", "duplicate node"),
            ("unnamed assignment", "unnamed"),
            ("operatorless assignment", "operatorless"),
            ("missing assignment", "missing or extra assignments"),
            ("extra assignment", "missing or extra assignments"),
            ("changed node name", "unexpected assigned node identity"),
            ("changed node domain", "unexpected assigned node identity"),
            ("changed node op", "unexpected assigned node identity"),
            ("CPU dequantization", "numerical node assigned to CPU"),
            ("CPU matrix multiply", "numerical node assigned to CPU"),
            ("unknown provider", "unrecognized assigned provider"),
            ("duplicate policy", "duplicate node"),
            ("missing policy", "missing or extra assignments"),
            ("extra policy", "missing or extra assignments"),
            ("numerical shape assertion", "unreviewed operator class"),
            ("custom domain shape assertion", "unreviewed operator class"),
            ("no numerical work", "no accelerated numerical work"),
        ] {
            let (mut policy, mut assignments) = reviewed_placement();
            let mut model_hash = "reviewed-graph-sha256";
            match attack {
                "different graph" => model_hash = "unreviewed-graph-sha256",
                "no assignments" => assignments.clear(),
                "duplicate assignment" => assignments.push(assignments[0].clone()),
                "unnamed assignment" => assignments[0].node.name.clear(),
                "operatorless assignment" => assignments[0].node.op.clear(),
                "missing assignment" => {
                    assignments.pop();
                }
                "extra assignment" => {
                    let mut extra = assignments[0].clone();
                    extra.node.name = "new-shape-node".into();
                    assignments.push(extra);
                }
                "changed node name" => assignments[0].node.name.push_str("-changed"),
                "changed node domain" => assignments[0].node.domain = "custom".into(),
                "changed node op" => assignments[0].node.op = "Cast".into(),
                "CPU dequantization" => assignments[1].provider = "CPUExecutionProvider".into(),
                "CPU matrix multiply" => assignments[2].provider = "CPUExecutionProvider".into(),
                "unknown provider" => assignments[1].provider = "OtherExecutionProvider".into(),
                "duplicate policy" => policy.nodes.push(policy.nodes[0].clone()),
                "missing policy" => {
                    policy.nodes.pop();
                }
                "extra policy" => {
                    let mut extra = policy.nodes[0].clone();
                    extra.node.name = "folded-away-shape".into();
                    policy.nodes.push(extra);
                }
                "numerical shape assertion" => policy.nodes[2].cpu_shape = true,
                "custom domain shape assertion" => {
                    policy.nodes[0].node.domain = "custom".into();
                    assignments[0].node.domain = "custom".into();
                }
                "no numerical work" => {
                    policy.nodes.truncate(1);
                    assignments.truncate(1);
                }
                _ => unreachable!(),
            }
            let error = admit_placement(&policy, model_hash, &assignments)
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected_error), "{attack}: {error}");
        }
    }

    #[test]
    fn rejects_truncated_negative_and_overflowing_tensor_shapes() {
        for (shape, bytes, width) in [
            (vec![1, 512], 2047, 4),
            (vec![-1, 2], 8, 4),
            (vec![i64::MAX, 2], 0, 8),
        ] {
            assert!(super::check_tensor(&shape, bytes, width).is_err());
        }
        super::check_tensor(&[1, 512], 2048, 4).unwrap();
        super::check_tensor(&[1, 0, 16], 0, 4).unwrap();
        super::check_tensor(&[], 8, 8).unwrap();
    }
}

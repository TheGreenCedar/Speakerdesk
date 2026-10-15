//! Native graph qualification; model-specific exporters own numerical acceptance.
use ort::{
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::Session,
    value::{DynTensorValueType, DynValue, Tensor},
};
use serde::Deserialize;
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
    #[serde(default)]
    dml_device_outputs: Vec<String>,
}

#[path = "../../../../crates/models/src/windows_placement.rs"]
mod windows_placement;
use windows_placement::{
    PlacementPolicy, admit_placement, placement_assignments, validate_assignment_table,
};

fn validate_device_outputs(
    provider: &str,
    adapter: i32,
    available: &[String],
    requested: &[String],
) -> Result<()> {
    if requested.is_empty() {
        return Ok(());
    }
    if provider != "dml" || adapter != 0 {
        return Err("device-output diagnostic requires DirectML physical adapter 0".into());
    }
    let mut seen = std::collections::BTreeSet::new();
    for name in requested {
        if !seen.insert(name) {
            return Err("duplicate device-output name".into());
        }
        if matches!(name.as_str(), "logits" | "encoder_mask") {
            return Err("logits and encoder_mask must remain CPU outputs".into());
        }
        if !available.contains(name) {
            return Err(format!("unknown device-output name: {name}").into());
        }
    }
    Ok(())
}

type OwnedOutputs = Vec<(String, DynValue)>;

fn run_bound(
    session: &mut Session,
    inputs: &[(String, DynValue)],
    device_outputs: &[String],
    options: &ort::session::RunOptions,
) -> Result<(OwnedOutputs, Vec<serde_json::Value>)> {
    let mut binding = session.create_binding()?;
    let dml = MemoryInfo::new(
        AllocationDevice::DIRECTML,
        0,
        AllocatorType::Device,
        MemoryType::Default,
    )?;
    let cpu = MemoryInfo::new(
        AllocationDevice::CPU,
        0,
        AllocatorType::Device,
        MemoryType::CPUOutput,
    )?;
    // Bind outputs first so even a partially failed input upload can be drained.
    let setup = (|| -> ort::Result<()> {
        for output in session.outputs() {
            binding.bind_output_to_device(
                output.name(),
                if device_outputs.iter().any(|n| n == output.name()) {
                    &dml
                } else {
                    &cpu
                },
            )?;
        }
        for (name, value) in inputs {
            binding.bind_input(name, value)?;
        }
        binding.synchronize_inputs()
    })();
    if let Err(cause) = setup {
        let drained = binding.synchronize_outputs();
        return Err(format!("binding setup failed: {cause}; output drain: {drained:?}").into());
    }
    let ran = session.run_binding_with_options(&binding, options);
    // DML OnRunEnd only flushes. Always wait, including error/cancellation paths.
    let drained = binding.synchronize_outputs();
    let outputs = match ran {
        Ok(outputs) => outputs,
        Err(cause) => {
            return Err(format!("bound Run failed: {cause}; output drain: {drained:?}").into());
        }
    };
    drained?;
    let mut owned = Vec::new();
    let mut locations = Vec::new();
    for (name, value) in outputs {
        let tensor = value.downcast_ref::<DynTensorValueType>()?;
        let info = tensor.memory_info();
        let expected = if device_outputs.iter().any(|n| n == name) {
            AllocationDevice::DIRECTML
        } else {
            AllocationDevice::CPU
        };
        if info.allocation_device() != expected || info.device_id() != 0 {
            return Err(format!("output {name} is not on its requested device").into());
        }
        locations.push(
            json!({"name": name, "allocation_device": info.allocation_device().as_str(),
            "device_id": info.device_id(), "memory_type": format!("{:?}", info.memory_type()),
            "device_type": format!("{:?}", info.device_type())}),
        );
        if name == "logits" {
            value.try_extract_tensor::<f32>()?;
        }
        owned.push((name.to_owned(), value));
    }
    // The returned values own their originating allocator; none borrows binding.
    Ok((owned, locations))
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
    let adapter: i32 = if provider == "dml" {
        args.get(5)
            .map(|value| value.to_string_lossy().parse())
            .transpose()?
            .unwrap_or(0)
    } else {
        0
    };
    let case_path = Path::new(&args[3]).canonicalize()?;
    let root = case_path.parent().ok_or("case has no parent")?;
    let case: Case = serde_json::from_slice(&fs::read(&case_path)?)?;
    if !(1..=1000).contains(&case.repeat) || case.inputs.is_empty() {
        return Err("case requires inputs and 1..=1000 repetitions".into());
    }
    if case.idle_ms > 5000 || (case.require_dml_only && provider != "dml") {
        return Err("idle must be <=5000ms and strict DirectML requires dml".into());
    }
    if !case.dml_device_outputs.is_empty() && (provider != "dml" || adapter != 0) {
        return Err("device-output diagnostic requires DirectML physical adapter 0".into());
    }
    let capture_placement = case.inspect_placement_only || case.placement_policy.is_some();
    if capture_placement
        && (provider != "dml" || case.require_dml_only || !case.disable_optimizations)
    {
        return Err("placement inspection/admission requires dml, disabled optimization, and no conflicting global strict flag".into());
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
    validate_device_outputs(
        provider,
        adapter,
        &session
            .outputs()
            .iter()
            .map(|v| v.name().to_owned())
            .collect::<Vec<_>>(),
        &case.dml_device_outputs,
    )?;
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
    let mut oracle_copy_ms = Vec::new();
    let mut serialization_ms = Vec::new();
    let mut total_ms = Vec::new();
    let mut output_locations = Vec::new();
    let binding_options = if case.dml_device_outputs.is_empty() {
        None
    } else {
        Some(ort::session::RunOptions::new()?)
    };
    for run in 0..case.repeat {
        run_phase(&output, run, false)?;
        let total_started = Instant::now();
        // Binding can queue uploads: the clock starts before creating/binding it.
        let started = Instant::now();
        let (outputs, elapsed) = if let Some(options) = &binding_options {
            let (values, locations) =
                run_bound(&mut session, &inputs, &case.dml_device_outputs, options)?;
            let elapsed = started.elapsed().as_secs_f64() * 1000.;
            output_locations.push(json!({"run": run, "outputs": locations}));
            (values, elapsed)
        } else {
            let outputs = session.run(
                inputs
                    .iter()
                    .map(|(name, value)| (name.as_str(), value))
                    .collect::<Vec<_>>(),
            )?;
            let elapsed = started.elapsed().as_secs_f64() * 1000.;
            (
                outputs
                    .into_iter()
                    .map(|(name, value)| (name.to_owned(), value))
                    .collect(),
                elapsed,
            )
        };
        run_ms.push(elapsed);
        run_phase(&output, run, true)?;
        // Oracle materialization is diagnostic only, after all GPU work completed.
        let copy_started = Instant::now();
        let outputs: OwnedOutputs = outputs
            .into_iter()
            .map(|(name, value)| -> Result<_> {
                if case.dml_device_outputs.contains(&name) {
                    let cpu = value
                        .downcast_ref::<DynTensorValueType>()?
                        .to(AllocationDevice::CPU, 0)?;
                    if cpu.memory_info().allocation_device() != AllocationDevice::CPU {
                        return Err("oracle copy did not return a CPU tensor".into());
                    }
                    Ok((name, cpu.into()))
                } else {
                    Ok((name, value))
                }
            })
            .collect::<Result<_>>()?;
        oracle_copy_ms.push(copy_started.elapsed().as_secs_f64() * 1000.);
        let serialization_started = Instant::now();
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
        serialization_ms.push(serialization_started.elapsed().as_secs_f64() * 1000.);
        total_ms.push(total_started.elapsed().as_secs_f64() * 1000.);
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
    let mut report = json!({
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
    if binding_options.is_some() {
        report["dml_device_outputs"] = json!(case.dml_device_outputs);
        report["output_locations_before_oracle_copy"] = json!(output_locations);
        report["oracle_copy_ms"] = json!(oracle_copy_ms);
        report["serialization_ms"] = json!(serialization_ms);
        report["total_ms"] = json!(total_ms);
        report["run_timing_scope"] = json!(
            "binding+input synchronization+Run+output synchronization+CPU logits ready; oracle copy and serialization separate"
        );
    }
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
    #[test]
    fn device_output_contract_rejects_wrong_provider_adapter_names_and_duplicates() {
        let available = [
            "present_self_k_0",
            "matrix0_product",
            "logits",
            "encoder_mask",
        ]
        .map(str::to_owned);
        super::validate_device_outputs("cpu", 7, &[], &[]).unwrap();
        super::validate_device_outputs("dml", 0, &available, &["matrix0_product".into()]).unwrap();
        for (provider, adapter, names, error) in [
            ("cpu", 0, vec!["present_self_k_0"], "physical adapter 0"),
            ("dml", 1, vec!["present_self_k_0"], "physical adapter 0"),
            ("dml", -1, vec!["present_self_k_0"], "physical adapter 0"),
            ("dml", 0, vec!["missing_cache"], "unknown"),
            (
                "dml",
                0,
                vec!["present_self_k_0", "present_self_k_0"],
                "duplicate",
            ),
            ("dml", 0, vec!["logits"], "CPU outputs"),
            ("dml", 0, vec!["encoder_mask"], "CPU outputs"),
        ] {
            let names = names.into_iter().map(str::to_owned).collect::<Vec<_>>();
            assert!(
                super::validate_device_outputs(provider, adapter, &available, &names)
                    .unwrap_err()
                    .to_string()
                    .contains(error)
            );
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

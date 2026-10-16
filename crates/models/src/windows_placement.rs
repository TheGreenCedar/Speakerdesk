//! Exact graph-bound DirectML placement admission shared with native qualification.
use ort::{AsPointer, session::Session};
use serde::{Deserialize, Serialize};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

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
pub(crate) struct PlacementPolicy {
    pub(crate) model_sha256: String,
    // Exact surviving assignment snapshot, reviewed against the pinned graph;
    // omissions for required folding/deletion are never inferred by the probe.
    nodes: Vec<PolicyNode>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Assignment {
    node: NodeIdentity,
    provider: String,
    subgraph: usize,
}

pub(crate) fn validate_assignment_table(assignments: &[Assignment]) -> Result<()> {
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

pub(crate) fn admit_placement(
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

pub(crate) fn placement_assignments(session: &Session) -> Result<Vec<Assignment>> {
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
}

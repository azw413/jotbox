use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Serialize)]
pub struct Node {
    pub id: String,
    pub name: String,
    pub group: u32,
    pub topic: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Link {
    pub source: String,
    pub target: String,
}

#[derive(Debug, Serialize)]
pub struct JsonGraph {
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
    #[serde(default)]
    pub pending_topics: usize,
}

impl JsonGraph {
    fn to_adjacency_graph(&self) -> (Graph, HashMap<usize, String>) {
        let mut next_id = 0;
        let mut mapping = HashMap::new();
        let mut reverse_mapping: HashMap<String, usize> = HashMap::new();

        // Create the mapping
        for n in &self.nodes {
            mapping.insert(next_id, n.id.clone());
            reverse_mapping.insert(n.id.clone(), next_id);
            next_id += 1;
        }

        let mut adj = Graph::new(next_id);

        for l in &self.links {
            let u = *reverse_mapping.get(&l.source).unwrap();
            let v = *reverse_mapping.get(&l.target).unwrap();

            adj.add_edge(u, v);
        }

        (adj, mapping)
    }

    pub fn assign_groups(&mut self) -> HashMap<String, u32> {
        let (adj, mapping) = self.to_adjacency_graph();
        let mut scc = kosaraju(&adj);

        scc.sort_by_key(|component| {
            component
                .iter()
                .filter_map(|node_idx| mapping.get(node_idx).and_then(|id| id.parse::<i64>().ok()))
                .min()
                .unwrap_or(i64::MAX)
        });

        let mut group_mapping: HashMap<String, u32> = HashMap::new();
        for (group_id, component) in scc.into_iter().enumerate() {
            for n in &component {
                group_mapping.insert(mapping.get(n).unwrap().to_string(), group_id as u32);
            }
        }

        for n in self.nodes.iter_mut() {
            if let Some(group) = group_mapping.get(&n.id) {
                n.group = *group;
            }
        }

        group_mapping
    }
}

// Kosaraju algorithm, a linear-time algorithm to find the strongly connected components (SCCs) of a directed graph, in Rust.
#[derive(Debug)]
pub struct Graph {
    vertices: usize,
    adj_list: Vec<Vec<usize>>,
    transpose_adj_list: Vec<Vec<usize>>,
}

impl Graph {
    pub fn new(vertices: usize) -> Self {
        Graph {
            vertices,
            adj_list: vec![vec![]; vertices],
            transpose_adj_list: vec![vec![]; vertices],
        }
    }

    pub fn add_edge(&mut self, u: usize, v: usize) {
        self.adj_list[u].push(v);
        self.transpose_adj_list[v].push(u);
    }

    pub fn dfs(&self, node: usize, visited: &mut Vec<bool>, stack: &mut Vec<usize>) {
        visited[node] = true;
        for &neighbor in &self.adj_list[node] {
            if !visited[neighbor] {
                self.dfs(neighbor, visited, stack);
            }
        }
        stack.push(node);
    }

    pub fn dfs_scc(&self, node: usize, visited: &mut Vec<bool>, scc: &mut Vec<usize>) {
        visited[node] = true;
        scc.push(node);
        for &neighbor in &self.transpose_adj_list[node] {
            if !visited[neighbor] {
                self.dfs_scc(neighbor, visited, scc);
            }
        }
    }
}

pub fn kosaraju(graph: &Graph) -> Vec<Vec<usize>> {
    let mut visited = vec![false; graph.vertices];
    let mut stack = Vec::new();

    for i in 0..graph.vertices {
        if !visited[i] {
            graph.dfs(i, &mut visited, &mut stack);
        }
    }

    let mut sccs = Vec::new();
    visited = vec![false; graph.vertices];

    while let Some(node) = stack.pop() {
        if !visited[node] {
            let mut scc = Vec::new();
            graph.dfs_scc(node, &mut visited, &mut scc);
            sccs.push(scc);
        }
    }

    sccs
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kosaraju_single_sccs() {
        let vertices = 5;
        let mut graph = Graph::new(vertices);

        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(2, 4);
        graph.add_edge(3, 0);
        graph.add_edge(4, 2);

        let sccs = kosaraju(&graph);
        assert_eq!(sccs.len(), 1);
        assert!(sccs.contains(&vec![0, 3, 2, 1, 4]));
    }

    #[test]
    fn test_kosaraju_multiple_sccs() {
        let vertices = 8;
        let mut graph = Graph::new(vertices);

        graph.add_edge(1, 0);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 0);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);
        graph.add_edge(4, 5);
        graph.add_edge(5, 6);
        graph.add_edge(6, 7);
        graph.add_edge(4, 7);
        graph.add_edge(6, 4);

        let sccs = kosaraju(&graph);
        assert_eq!(sccs.len(), 4);
        assert!(sccs.contains(&vec![0, 1, 2]));
        assert!(sccs.contains(&vec![3]));
        assert!(sccs.contains(&vec![4, 6, 5]));
        assert!(sccs.contains(&vec![7]));
    }

    #[test]
    fn test_kosaraju_multiple_sccs1() {
        let vertices = 8;
        let mut graph = Graph::new(vertices);
        graph.add_edge(0, 2);
        graph.add_edge(1, 0);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);
        graph.add_edge(4, 7);
        graph.add_edge(5, 2);
        graph.add_edge(5, 6);
        graph.add_edge(6, 5);
        graph.add_edge(7, 6);

        let sccs = kosaraju(&graph);
        assert_eq!(sccs.len(), 3);
        assert!(sccs.contains(&vec![0]));
        assert!(sccs.contains(&vec![1]));
        assert!(sccs.contains(&vec![2, 5, 6, 7, 4, 3]));
    }

    #[test]
    fn test_kosaraju_no_scc() {
        let vertices = 4;
        let mut graph = Graph::new(vertices);

        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);

        let sccs = kosaraju(&graph);
        assert_eq!(sccs.len(), 4);
        for (i, _) in sccs.iter().enumerate().take(vertices) {
            assert_eq!(sccs[i], vec![i]);
        }
    }

    #[test]
    fn test_kosaraju_empty_graph() {
        let vertices = 0;
        let graph = Graph::new(vertices);

        let sccs = kosaraju(&graph);
        assert_eq!(sccs.len(), 0);
    }
}

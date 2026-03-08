use ark_serialize::CanonicalSerialize;
use terkle::{MultiverkleTree, Scalar, TreeConfig};
use rand::{rngs::StdRng, SeedableRng};

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let config = TreeConfig {
        axis_arity: vec![4, 4],
        depth: 2,
    };
    let total = config.axis_product().pow(config.depth as u32);
    let data = (0..total)
        .map(|i| Scalar::from((i + 1) as u64))
        .collect::<Vec<_>>();

    let tree = MultiverkleTree::from_data(&mut rng, config.clone(), data.clone())
        .expect("tree construction should succeed");
    let mut root_bytes = Vec::new();
    tree.root_commitment()
        .serialize_compressed(&mut root_bytes)
        .expect("serialize root");
    println!(
        "Root commitment (compressed hex): 0x{}",
        hex::encode(&root_bytes)
    );

    let index = 53;
    let claimed_value = data[index];
    let proof = tree
        .open_linear_index(index)
        .expect("proof should build for index");

    // Verifier without access to the tree only needs public parameters.
    let without_tree = MultiverkleTree::verify_membership(
        &config,
        index,
        &claimed_value,
        tree.verification_key(),
        tree.root_commitment(),
        &proof,
    )
    .expect("verification should not error");
    println!("Verification without tree: {}", without_tree);

    // Sanity-check using the tree helper as well.
    let with_tree = tree
        .verify_membership_with_tree(index, &claimed_value, &proof)
        .expect("tree verification should not error");
    println!("Verification with tree helper: {}", with_tree);

    assert!(without_tree && with_tree);
    println!("Proof verified! Leaf value: {}", proof.leaf_value);
}

use shader_reflection::SpirVModule;

fn main() {
    let file_path = match std::env::args().nth(1) {
        Some(it) => it,
        None => panic!("inspected file expected as first argument")
    };

    let module = SpirVModule::open(file_path).unwrap();

    dbg!(module.metadata);
}

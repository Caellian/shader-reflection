use shader_reflection::{SpirVModule, codegen::TokenizationState};

fn main() {
    let file_path = match std::env::args().nth(1) {
        Some(it) => it,
        None => panic!("inspected file expected as first argument")
    };
    
    let module = SpirVModule::open(file_path).unwrap();

    let mut state = TokenizationState::new(&module.metadata.types);
    
    let mut items = Vec::new();
    for entry_point in &module.metadata.entry_points {
        items.push(state.entry_point_module(entry_point));
    }

    let generated = syn::File {
        shebang: None,
        attrs: vec![],
        items,
    };

    let content = prettyplease::unparse(&generated);
    println!("{}", content);
}

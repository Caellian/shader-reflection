use std::collections::{HashMap, HashSet};

use proc_macro2::{Ident, Span};
use syn::{parse_quote, punctuated::Punctuated, Item, PathSegment, TypePath, Token, token, TypeTuple};

use crate::{types::{SpirvLiteralType, SpirvStructField, SpirvType, SpirvTypeOwned}, EntryPoint};

pub struct TokenizationState<'i> {
    type_info: &'i HashMap<u32, SpirvTypeOwned>,
}
impl<'i> TokenizationState<'i> {
    pub fn new(type_info: &'i HashMap<u32, SpirvTypeOwned>) -> Self {
        TokenizationState { type_info }
    }

    pub fn produce_items(&mut self, for_types: Vec<u32>) -> Vec<Item> {
        let mut produced = HashMap::new();
        let mut remaining = for_types;
        while let Some(current) = remaining.pop() {
            if produced.contains_key(&current) {
                continue;
            }
            let t = self
                .type_info
                .get(&current)
                .expect("required type not found");
            let mut pass = TokenizationPass::new(self.type_info);
            
            let item = pass.tokenize_item(t);
            remaining.append(&mut pass.requirements);
            if let Some(item) = item {
                produced.insert(current, item);
            }
        }

        produced.into_values().collect()
    }

    pub fn entry_point_module(&mut self, entry_point: &EntryPoint) -> syn::Item {
        let ident = entry_point.name.to_string_lossy().to_string();
        let ident = Ident::new(&ident, Span::call_site());

        let mut items: Vec<syn::Item> = Vec::new();
        let mut required_types = HashSet::new();
        let mut naming_pass = TokenizationPass::new(self.type_info);
        for binding in &entry_point.bindings {
            let name = match &binding.name {
                Some(it) => it.clone(),
                None => format!("BINDING_{}_{}", binding.set, binding.binding)
            };
            let name = Ident::new(&name, Span::call_site());
            let ty_name = self.type_info.get(&binding.spirv_type_ref).expect("tokenizer missing binding type");
            let ty_name = naming_pass.tokenize_type_name(ty_name);
            items.push(parse_quote!{
                pub type #name = #ty_name;
            });
            required_types.insert(binding.spirv_type_ref);
        }
        if let Some(push_constant) = &entry_point.push_constant {
            let name = match &push_constant.name {
                Some(it) => it.clone(),
                None => String::from("PUSH_CONSTANT"),
            };
            let name = Ident::new(&name, Span::call_site());
            let ty_name = self.type_info.get(&push_constant.spirv_type_ref).expect("tokenizer missing push constant type");
            let ty_name = naming_pass.tokenize_type_name(ty_name);
            items.push(parse_quote!{
                pub type #name = #ty_name;
            });
            required_types.insert(push_constant.spirv_type_ref);
        }
        items.append(&mut self.produce_items(required_types.into_iter().collect()));

        parse_quote!{
            pub mod #ident {
                #(#items)*
            }
        }
    }
}

struct TokenizationPass<'i> {
    type_info: &'i HashMap<u32, SpirvTypeOwned>,
    variable_arrays: Vec<String>,
    array_names: HashMap<u32, String>,
    requirements: Vec<u32>,
}
impl<'i> TokenizationPass<'i> {
    fn new(type_info: &'i HashMap<u32, SpirvTypeOwned>) -> Self {
        Self {
            type_info,
            variable_arrays: Vec::new(),
            array_names: HashMap::new(),
            requirements: Vec::new(),
        }
    }
    fn tokenize_item(&mut self, t: &SpirvTypeOwned) -> Option<syn::Item> {
        Some(match t {
            SpirvTypeOwned::Struct(it) => {
                if it.interface {
                    return self
                        .tokenize_item(&it.fields.first().expect("empty interface struct").ty);
                }

                let name = it
                    .name()
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("Struct{}", it.type_reference()));
                let name = Ident::new(&name, Span::call_site());
                let mut fields = Vec::new();
                for (i, inner) in it.fields.iter().enumerate() {
                    if let Some(padding) = it.padding_before(i) {
                        let name = format!("_padding_before_{}", i);
                        let name = Ident::new(&name, Span::call_site());
                        let padding = parse_quote! {
                            #name: [u8; #padding]
                        };
                        fields.push(padding)
                    }
                    fields.push(self.tokenize_field(inner, i))
                }
                parse_quote! {
                    #[repr(C)]
                    pub struct #name<> {
                        #(#fields),*
                    }
                }
            }
            SpirvTypeOwned::Pointer(it) => self.tokenize_item(it.ty.as_ref())?,
            other => return None,
        })
    }

    fn tokenize_field(&mut self, field: &SpirvStructField, index: usize) -> syn::Field {
        let name = match &field.name {
            Some(it) => it.clone(),
            None => format!("field_{}", index),
        };
        let name = Ident::new(&name, Span::call_site());
        let ty = self.tokenize_type_name(&field.ty);
        parse_quote! {
            pub #name: #ty
        }
    }

    fn tokenize_type_name(&mut self, ty: &SpirvTypeOwned) -> syn::Type {
        match ty {
            SpirvTypeOwned::Struct(it) => {
                if it.interface {
                    return self
                        .tokenize_type_name(&it.fields.first().expect("empty interface struct").ty);
                }

                self.requirements.push(it.type_reference());

                let name = it
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("Struct{}", it.type_reference()));
                syn::Type::Path(TypePath {
                    qself: None,
                    path: syn::Path::from(Ident::new(&name, Span::call_site())),
                })
            }
            SpirvTypeOwned::Array(it) => {
                let ty = self.tokenize_type_name(it.value_type.as_ref());
                match it.length {
                    Some(length) => {
                        parse_quote! {
                            [#ty; #length]
                        }
                    }
                    None => {
                        let size_name = self
                            .array_names
                            .get(&it.type_reference())
                            .map(|it| it.to_ascii_uppercase())
                            .unwrap_or_default();
                        parse_quote! {
                            [#ty; #size_name]
                        }
                    }
                }
            }
            SpirvTypeOwned::Vector(it) => {
                #[cfg(feature = "glam")]
                if (2..=4).contains(&it.width) {
                    let prefix = match it.component_type.as_ref() {
                        SpirvTypeOwned::Literal(literal) => match literal {
                            crate::types::SpirvLiteralType::Float { width: 32, .. } => Some(""),
                            crate::types::SpirvLiteralType::Float { width: 64, .. } => Some("D"),
                            crate::types::SpirvLiteralType::Integer {
                                width: 16,
                                signed: false,
                                ..
                            } => Some("U16"),
                            crate::types::SpirvLiteralType::Integer {
                                width: 16,
                                signed: true,
                                ..
                            } => Some("I16"),
                            crate::types::SpirvLiteralType::Integer {
                                width: 32,
                                signed: false,
                                ..
                            } => Some("U"),
                            crate::types::SpirvLiteralType::Integer {
                                width: 32,
                                signed: true,
                                ..
                            } => Some("I"),
                            crate::types::SpirvLiteralType::Integer {
                                width: 64,
                                signed: false,
                                ..
                            } => Some("U64"),
                            crate::types::SpirvLiteralType::Integer {
                                width: 64,
                                signed: true,
                                ..
                            } => Some("I64"),
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(prefix) = prefix {
                        return glam_type(format!("{}Vec{}", prefix, it.width));
                    }
                }

                self.tokenize_type_name(&SpirvTypeOwned::Array(it.to_array()))
            }
            SpirvTypeOwned::Matrix(it) => {
                // TODO: Handle stride
                let columns = it.columns;
                let rows = it.column_type.width;

                #[cfg(feature = "glam")]
                if (2..=4).contains(&columns) {
                    let prefix = match it.column_type.component_type.as_ref() {
                        SpirvTypeOwned::Literal(literal) => match literal {
                            crate::types::SpirvLiteralType::Float { width: 32, .. } => Some(""),
                            crate::types::SpirvLiteralType::Float { width: 64, .. } => Some("D"),
                            _ => None,
                        },
                        _ => None,
                    };
                    if let Some(prefix) = prefix {
                        if columns == rows {
                            return glam_type(format!("{}Mat{}", prefix, columns));
                        } else if !it.is_row_major() && rows == columns - 1 && columns >= 3 {
                            return glam_type(format!("{}Affine{}", prefix, rows));
                        }
                    }
                }

                let item = self.tokenize_type_name(it.column_type.component_type.as_ref());

                if it.is_row_major() {
                    parse_quote! {
                        [[#item; #rows]; #columns]
                    }
                } else {
                    parse_quote! {
                        [[#item; #columns]; #rows]
                    }
                }
            }
            SpirvTypeOwned::Literal(it) => {
                let ident = match it {
                    SpirvLiteralType::Float { width: 32, .. } => "f32",
                    SpirvLiteralType::Float { width: 64, .. } => "f64",
                    SpirvLiteralType::Integer {
                        width: 8,
                        signed: false,
                        ..
                    } => "u8",
                    SpirvLiteralType::Integer {
                        width: 8,
                        signed: true,
                        ..
                    } => "i8",
                    SpirvLiteralType::Integer {
                        width: 16,
                        signed: false,
                        ..
                    } => "u16",
                    SpirvLiteralType::Integer {
                        width: 16,
                        signed: true,
                        ..
                    } => "i16",
                    SpirvLiteralType::Integer {
                        width: 32,
                        signed: false,
                        ..
                    } => "u32",
                    SpirvLiteralType::Integer {
                        width: 32,
                        signed: true,
                        ..
                    } => "i32",
                    SpirvLiteralType::Integer {
                        width: 64,
                        signed: false,
                        ..
                    } => "u64",
                    SpirvLiteralType::Integer {
                        width: 64,
                        signed: true,
                        ..
                    } => "i64",
                    SpirvLiteralType::Integer {
                        width: 128,
                        signed: false,
                        ..
                    } => "u128",
                    SpirvLiteralType::Integer {
                        width: 128,
                        signed: true,
                        ..
                    } => "i128",
                    other => panic!("unsupported type: {:?}", other),
                };
                syn::Type::Path(TypePath {
                    qself: None,
                    path: syn::Path::from(Ident::new(ident, Span::call_site())),
                })
            }
            SpirvTypeOwned::Pointer(it) => self.tokenize_type_name(it.ty.as_ref()),
            _ => syn::Type::Tuple(TypeTuple {
                paren_token: token::Paren::default(),
                elems: Punctuated::new(),
            })
        }
    }
}

#[cfg(feature = "glam")]
fn glam_type(name: impl AsRef<str>) -> syn::Type {
    syn::Type::Path(TypePath {
        qself: None,
        path: syn::Path {
            leading_colon: None,
            segments: Punctuated::from_iter([
                PathSegment {
                    ident: Ident::new("glam", Span::call_site()),
                    arguments: syn::PathArguments::None,
                },
                PathSegment {
                    ident: Ident::new(name.as_ref(), Span::call_site()),
                    arguments: syn::PathArguments::None,
                },
            ]),
        },
    })
}

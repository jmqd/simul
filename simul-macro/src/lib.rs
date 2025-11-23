use proc_macro::{Span, TokenStream};
use quote::quote;
use syn::{self, token, Field, Ident, Visibility};

#[proc_macro_attribute]
pub fn agent(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut agent_struct = syn::parse_macro_input!(item as syn::ItemStruct);
    let _ = syn::parse_macro_input!(attr as syn::parse::Nothing);

    if let syn::Fields::Named(ref mut fields) = agent_struct.fields {
        if fields
            .named
            .iter()
            .all(|f| f.ident.as_ref().map(|name| name != "id").unwrap_or(false))
        {
            fields.named.push(Field {
                attrs: vec![],
                vis: Visibility::Inherited,
                ident: Some(Ident::new("id", Span::call_site().into())),
                colon_token: Some(token::Colon {
                    spans: [Span::call_site().into()],
                }),
                ty: syn::parse_str("String").expect("Failed to parse type"),
                mutability: syn::FieldMutability::None,
            });
        }
    }

    quote!(
        #[derive(Clone, Debug)]
        #agent_struct
    )
    .into()
}

use proc_macro::{Span, TokenStream};
use quote::quote;
use syn::{self, token, Field, Ident, Visibility};

#[proc_macro_attribute]
pub fn agent(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut agent_struct = syn::parse_macro_input!(item as syn::ItemStruct);
    let _ = syn::parse_macro_input!(attr as syn::parse::Nothing);

    if let syn::Fields::Named(ref mut fields) = agent_struct.fields {
        if fields.named.iter().all(|f| {
            f.ident
                .as_ref()
                .map(|name| name != "state")
                .unwrap_or(false)
        }) {
            fields.named.push(Field {
                attrs: vec![],
                vis: Visibility::Inherited,
                ident: Some(Ident::new("state", Span::call_site().into())),
                colon_token: Some(token::Colon {
                    spans: [Span::call_site().into()],
                }),
                ty: syn::parse_str("simul::AgentState").expect("Failed to parse type"),
                mutability: syn::FieldMutability::None,
            });
        }
    }

    let struct_name = &agent_struct.ident;
    let common_agent_impl = quote! {
        impl simul::AgentCommon for #struct_name {
            fn state(&self) -> &AgentState {
                &self.state
            }

            fn state_mut(&mut self) -> &mut AgentState {
                &mut self.state
            }
        }
    };

    quote!(
        #[derive(Clone, Debug)]
        #agent_struct

        #common_agent_impl
    )
    .into()
}

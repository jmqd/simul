use proc_macro::{Span, TokenStream};
use quote::quote;
use syn::{self, token, Field, Ident, Visibility};

#[proc_macro_attribute]
pub fn agent(attr: TokenStream, item: TokenStream) -> TokenStream {
    let agent_struct = syn::parse_macro_input!(item as syn::ItemStruct);

    quote!(
        #[derive(Clone, Debug)]
        #agent_struct
    )
    .into()
}

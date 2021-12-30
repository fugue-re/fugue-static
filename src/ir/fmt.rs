use fugue::ir::Translator;

pub struct DisplayFull<'t> {
    pub translator: Option<&'t Translator>,
    pub branch_start: &'t str,
    pub branch_end: &'t str,
    pub keyword_start: &'t str,
    pub keyword_end: &'t str,
    pub location_start: &'t str,
    pub location_end: &'t str,
    pub type_start: &'t str,
    pub type_end: &'t str,
    pub value_start: &'t str,
    pub value_end: &'t str,
    pub variable_start: &'t str,
    pub variable_end: &'t str,
}

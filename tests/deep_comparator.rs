//! Ignored integration test that wires the code-complexity-comparator
//! into the repo as an executable parity gate.

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Output;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn comparator_bin() -> PathBuf {
    std::env::var_os("ZSTD_PURE_RS_CCC_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "/data/henriksson/github/claude/code-complexity-comparator/target/release/ccc-rs",
            )
        })
}

fn workdir() -> PathBuf {
    std::env::var_os("ZSTD_PURE_RS_CCC_WORKDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("target").join("deep-comparator"))
}

fn run_checked(command: &mut Command, label: &str) -> Output {
    let out = command
        .output()
        .unwrap_or_else(|e| panic!("run {label}: {e}"));
    assert!(
        out.status.success(),
        "{label} failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

fn assert_nonempty_file(path: &Path, label: &str) {
    let metadata = fs::metadata(path)
        .unwrap_or_else(|e| panic!("{label} did not produce {}: {e}", path.display()));
    assert!(
        metadata.is_file() && metadata.len() > 0,
        "{label} produced an empty or non-file report at {}",
        path.display()
    );
}

fn assert_analysis_report_has_functions(path: &Path, label: &str, expected_language: &str) {
    let report = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{label} report {} was not readable: {e}", path.display()));
    assert_analysis_report_contents_has_functions(&report, label, Some(expected_language));
}

fn assert_analysis_report_contents_has_functions(
    report: &str,
    label: &str,
    expected_language: Option<&str>,
) {
    let json = parse_json(&report, &format!("{label} report"));
    let schema_version_value = json_object_field(&json, "schema_version", report)
        .unwrap_or_else(|| panic!("missing top-level JSON field schema_version in:\n{report}"));
    let JsonValue::Number(schema_version_text) = schema_version_value else {
        panic!("{label} report contained non-integer schema_version");
    };
    let schema_version: u32 = schema_version_text
        .parse()
        .unwrap_or_else(|_| panic!("{label} report contained non-integer schema_version"));
    assert_eq!(
        schema_version, 4,
        "{label} report used unexpected schema_version {schema_version}"
    );
    let language = json_object_string_field(&json, "language", report)
        .unwrap_or_else(|| panic!("{label} report contained no language"));
    assert_nonblank(language, || {
        format!("{label} report contained an empty language")
    });
    if let Some(expected_language) = expected_language {
        assert_eq!(
            language, expected_language,
            "{label} report used unexpected language"
        );
    }
    let source_file = json_object_string_field(&json, "source_file", report)
        .unwrap_or_else(|| panic!("{label} report contained no source_file"));
    assert_nonblank(source_file, || {
        format!("{label} report contained an empty source_file")
    });
    let source_hash = json_object_string_field(&json, "source_hash", report)
        .unwrap_or_else(|| panic!("{label} report contained no source_hash"));
    assert_empty_or_nonblank(source_hash, || {
        format!("{label} report contained a blank source_hash")
    });
    let functions = json_object_array_field(&json, "functions", &report);
    assert!(
        !functions.is_empty(),
        "{label} report contained no analyzed functions"
    );
    for (index, function) in functions.iter().enumerate() {
        assert_analysis_entry_schema(function, index, label, report);
    }
    if let Some(structs) = json_object_array_field_optional(&json, "structs", report) {
        for (index, struct_entry) in structs.iter().enumerate() {
            assert_struct_entry_schema(struct_entry, index, label, report);
        }
    }
}

fn assert_analysis_entry_schema(function: &JsonValue, index: usize, label: &str, report: &str) {
    let JsonValue::Object(_) = function else {
        panic!("{label} report contained a non-object function entry {index}");
    };
    let name = json_object_string_field(function, "name", report)
        .unwrap_or_else(|| panic!("{label} report contained a function without a name"));
    assert_nonblank(name, || {
        format!("{label} report contained a function with an empty name")
    });
    assert_optional_nonblank_string_field(function, "original_name", label, report);
    assert_optional_nonblank_string_field(function, "mangled", label, report);
    assert_optional_nonblank_string_field(function, "enclosing_type", label, report);

    let location = json_object_field(function, "location", report)
        .unwrap_or_else(|| panic!("{label} report function {name} contained no location"));
    let JsonValue::Object(_) = location else {
        panic!("{label} report function {name} contained non-object location");
    };
    let file = json_object_string_field(location, "file", report)
        .unwrap_or_else(|| panic!("{label} report function {name} contained no location file"));
    assert_nonblank(file, || {
        format!("{label} report function {name} contained an empty location file")
    });
    let line_start = json_object_u64_field(location, "line_start", report);
    let line_end = json_object_u64_field(location, "line_end", report);
    json_object_u64_field(location, "col_start", report);
    json_object_u64_field(location, "col_end", report);
    let byte_start = json_object_u64_field(location, "byte_start", report);
    let byte_end = json_object_u64_field(location, "byte_end", report);
    assert!(
        line_start <= line_end,
        "{label} report function {name} had inverted line span"
    );
    assert!(
        byte_start <= byte_end,
        "{label} report function {name} had inverted byte span"
    );

    let metrics = json_object_field(function, "metrics", report)
        .unwrap_or_else(|| panic!("{label} report function {name} contained no metrics"));
    let JsonValue::Object(fields) = metrics else {
        panic!("{label} report function {name} contained non-object metrics");
    };
    assert!(
        !fields.is_empty(),
        "{label} report function {name} contained empty metrics"
    );
    assert!(
        count_finite_numbers(metrics, report) > 0,
        "{label} report function {name} contained no numeric metrics"
    );
    assert_metrics_schema(metrics, name, label, report);
    assert_signature_schema(function, name, label, report);
    for (entry_index, constant) in json_object_array_field(function, "constants", report)
        .iter()
        .enumerate()
    {
        assert_constant_schema(constant, entry_index, label, report);
    }
    for (entry_index, call) in json_object_array_field(function, "calls", report)
        .iter()
        .enumerate()
    {
        assert_call_schema(call, entry_index, label, report);
    }
    if let Some(call_sites) = json_object_array_field_optional(function, "call_sites", report) {
        for (entry_index, call_site) in call_sites.iter().enumerate() {
            assert_call_site_schema(call_site, entry_index, label, report);
        }
    }
    for (entry_index, ty) in json_object_array_field(function, "types_used", report)
        .iter()
        .enumerate()
    {
        assert_type_ref_schema(ty, entry_index, "types_used", label, report);
    }
    assert_attributes_schema(function, label, report);
}

fn assert_struct_entry_schema(struct_entry: &JsonValue, index: usize, label: &str, report: &str) {
    let JsonValue::Object(_) = struct_entry else {
        panic!("{label} report contained a non-object struct entry {index}");
    };
    let name = json_object_string_field(struct_entry, "name", report)
        .unwrap_or_else(|| panic!("{label} report contained a struct without a name"));
    assert_nonblank(name, || {
        format!("{label} report contained a struct with an empty name")
    });
    let kind = json_object_string_field(struct_entry, "kind", report)
        .unwrap_or_else(|| panic!("{label} report struct {name} contained no kind"));
    assert_nonblank(kind, || {
        format!("{label} report struct {name} contained an empty kind")
    });
    assert!(
        matches!(
            kind,
            "struct" | "class" | "union" | "record" | "derived_type"
        ),
        "{label} report struct {name} used unknown kind {kind}"
    );

    assert_location_schema(struct_entry, "struct", name, label, report);

    let fields = json_object_array_field(struct_entry, "fields", report);
    let mut category_counts = StructCategoryCounts::default();
    for (field_index, field_entry) in fields.iter().enumerate() {
        let category = assert_struct_field_schema(field_entry, field_index, label, report);
        category_counts.increment(category);
    }
    assert_struct_metrics_schema(
        struct_entry,
        name,
        fields.len(),
        &category_counts,
        label,
        report,
    );
    assert_attributes_schema(struct_entry, label, report);
}

fn assert_struct_field_schema<'a>(
    field_entry: &'a JsonValue,
    index: usize,
    label: &str,
    report: &str,
) -> &'a str {
    let JsonValue::Object(_) = field_entry else {
        panic!("{label} report contained non-object struct field entry {index}");
    };
    let name = json_object_string_field(field_entry, "name", report)
        .unwrap_or_else(|| panic!("{label} report struct field entry {index} contained no name"));
    assert_nonblank(name, || {
        format!("{label} report struct field entry {index} contained an empty name")
    });
    let ty = json_object_field(field_entry, "ty", report)
        .unwrap_or_else(|| panic!("{label} report struct field entry {index} contained no type"));
    assert_type_ref_schema(ty, index, "struct field type", label, report);
    let category = json_object_string_field(field_entry, "category", report).unwrap_or_else(|| {
        panic!("{label} report struct field entry {index} contained no category")
    });
    assert!(
        matches!(
            category,
            "int"
                | "float"
                | "bool"
                | "char"
                | "string"
                | "pointer"
                | "array"
                | "collection"
                | "other"
        ),
        "{label} report struct field entry {index} used unknown category {category}"
    );
    category
}

#[derive(Default)]
struct StructCategoryCounts {
    int: u64,
    float: u64,
    bool: u64,
    char: u64,
    string: u64,
    pointer: u64,
    array: u64,
    collection: u64,
    other: u64,
}

impl StructCategoryCounts {
    fn increment(&mut self, category: &str) {
        match category {
            "int" => self.int += 1,
            "float" => self.float += 1,
            "bool" => self.bool += 1,
            "char" => self.char += 1,
            "string" => self.string += 1,
            "pointer" => self.pointer += 1,
            "array" => self.array += 1,
            "collection" => self.collection += 1,
            "other" => self.other += 1,
            _ => unreachable!("struct field category was already validated"),
        }
    }

    fn expected_metric(&self, field: &str) -> u64 {
        match field {
            "int_count" => self.int,
            "float_count" => self.float,
            "bool_count" => self.bool,
            "char_count" => self.char,
            "string_count" => self.string,
            "pointer_count" => self.pointer,
            "array_count" => self.array,
            "collection_count" => self.collection,
            "other_count" => self.other,
            _ => unreachable!("unknown struct metric count field"),
        }
    }
}

fn assert_struct_metrics_schema(
    struct_entry: &JsonValue,
    name: &str,
    field_count: usize,
    category_counts: &StructCategoryCounts,
    label: &str,
    report: &str,
) {
    let metrics = json_object_field(struct_entry, "metrics", report)
        .unwrap_or_else(|| panic!("{label} report struct {name} contained no metrics"));
    let JsonValue::Object(_) = metrics else {
        panic!("{label} report struct {name} contained non-object metrics");
    };
    let actual_field_count = json_object_u64_field(metrics, "field_count", report);
    assert_eq!(
        actual_field_count, field_count as u64,
        "{label} report struct {name} field_count did not match fields"
    );
    let category_fields = [
        "int_count",
        "float_count",
        "bool_count",
        "char_count",
        "string_count",
        "pointer_count",
        "array_count",
        "collection_count",
        "other_count",
    ];
    let mut category_total = 0;
    for field in category_fields {
        let actual = json_object_u64_field(metrics, field, report);
        let expected = category_counts.expected_metric(field);
        assert_eq!(
            actual, expected,
            "{label} report struct {name} {field} did not match fields"
        );
        category_total += actual;
    }
    assert_eq!(
        category_total, actual_field_count,
        "{label} report struct {name} category counts did not match field_count"
    );
}

fn assert_location_schema(
    json: &JsonValue,
    owner_kind: &str,
    owner_name: &str,
    label: &str,
    report: &str,
) {
    let location = json_object_field(json, "location", report).unwrap_or_else(|| {
        panic!("{label} report {owner_kind} {owner_name} contained no location")
    });
    let JsonValue::Object(_) = location else {
        panic!("{label} report {owner_kind} {owner_name} contained non-object location");
    };
    let file = json_object_string_field(location, "file", report).unwrap_or_else(|| {
        panic!("{label} report {owner_kind} {owner_name} contained no location file")
    });
    assert_nonblank(file, || {
        format!("{label} report {owner_kind} {owner_name} contained an empty location file")
    });
    let line_start = json_object_u64_field(location, "line_start", report);
    let line_end = json_object_u64_field(location, "line_end", report);
    json_object_u64_field(location, "col_start", report);
    json_object_u64_field(location, "col_end", report);
    let byte_start = json_object_u64_field(location, "byte_start", report);
    let byte_end = json_object_u64_field(location, "byte_end", report);
    assert!(
        line_start <= line_end,
        "{label} report {owner_kind} {owner_name} had inverted line span"
    );
    assert!(
        byte_start <= byte_end,
        "{label} report {owner_kind} {owner_name} had inverted byte span"
    );
}

fn assert_signature_schema(function: &JsonValue, name: &str, label: &str, report: &str) {
    let signature = json_object_field(function, "signature", report)
        .unwrap_or_else(|| panic!("{label} report function {name} contained no signature"));
    let JsonValue::Object(_) = signature else {
        panic!("{label} report function {name} contained non-object signature");
    };
    json_object_array_field(signature, "inputs", report);
    for (index, input) in json_object_array_field(signature, "inputs", report)
        .iter()
        .enumerate()
    {
        assert_param_schema(input, index, label, report);
    }
    for (index, output) in json_object_array_field(signature, "outputs", report)
        .iter()
        .enumerate()
    {
        assert_type_ref_schema(output, index, "signature outputs", label, report);
    }
}

fn assert_metrics_schema(metrics: &JsonValue, name: &str, label: &str, report: &str) {
    for field in [
        "loc_code",
        "loc_comments",
        "loc_asm",
        "inputs",
        "outputs",
        "branches",
        "loops",
        "max_loop_nesting",
        "max_if_nesting",
        "max_combined_nesting",
        "calls_unique",
        "calls_total",
        "cyclomatic",
        "cognitive",
        "early_returns",
        "goto_count",
        "unsafe_blocks",
    ] {
        json_object_u64_field(metrics, field, report);
    }

    let halstead = json_object_field(metrics, "halstead", report)
        .unwrap_or_else(|| panic!("{label} report function {name} contained no halstead metrics"));
    let JsonValue::Object(_) = halstead else {
        panic!("{label} report function {name} contained non-object halstead metrics");
    };
    for field in ["n1", "n2", "big_n1", "big_n2", "volume", "difficulty"] {
        nonnegative_json_number_field(halstead, field, report);
    }

    let binary_operators =
        json_object_field(metrics, "binary_operators", report).unwrap_or_else(|| {
            panic!("{label} report function {name} contained no binary operator metrics")
        });
    let JsonValue::Object(_) = binary_operators else {
        panic!("{label} report function {name} contained non-object binary operator metrics");
    };
    for field in [
        "add",
        "sub",
        "mul",
        "div",
        "rem",
        "shl",
        "shr",
        "bit_and",
        "bit_or",
        "bit_xor",
        "bit_not",
        "logic_and",
        "logic_or",
        "logic_not",
    ] {
        json_object_u64_field(binary_operators, field, report);
    }
}

fn assert_param_schema(param: &JsonValue, index: usize, label: &str, report: &str) {
    let JsonValue::Object(_) = param else {
        panic!("{label} report contained non-object signature input {index}");
    };
    let name = json_object_string_field(param, "name", report)
        .unwrap_or_else(|| panic!("{label} report signature input {index} contained no name"));
    assert_nonblank(name, || {
        format!("{label} report signature input {index} contained an empty name")
    });
    let ty = json_object_field(param, "ty", report)
        .unwrap_or_else(|| panic!("{label} report signature input {index} contained no type"));
    assert_type_ref_schema(ty, index, "signature input type", label, report);
}

fn assert_type_ref_schema(ty: &JsonValue, index: usize, field: &str, label: &str, report: &str) {
    let JsonValue::Object(_) = ty else {
        panic!("{label} report contained non-object {field} entry {index}");
    };
    let text = json_object_string_field(ty, "text", report)
        .unwrap_or_else(|| panic!("{label} report {field} entry {index} contained no text"));
    assert_nonblank(text, || {
        format!("{label} report {field} entry {index} contained empty text")
    });
}

fn assert_constant_schema(constant: &JsonValue, index: usize, label: &str, report: &str) {
    let JsonValue::Object(_) = constant else {
        panic!("{label} report contained non-object constant entry {index}");
    };
    let kind = json_object_string_field(constant, "kind", report)
        .unwrap_or_else(|| panic!("{label} report constant entry {index} contained no kind"));
    match kind {
        "int" => {
            json_object_i64_field(constant, "value", report);
            assert_nonempty_string_field(constant, "text", label, report);
        }
        "float" => {
            finite_json_number_field(constant, "value", report);
            assert_nonempty_string_field(constant, "text", label, report);
        }
        "string" => {
            json_object_string_field(constant, "value", report).unwrap_or_else(|| {
                panic!("{label} report constant entry {index} contained no string value")
            });
        }
        "char" => {
            let value = json_object_string_field(constant, "value", report).unwrap_or_else(|| {
                panic!("{label} report constant entry {index} contained no char value")
            });
            assert_nonempty(value, || {
                format!("{label} report constant entry {index} contained empty char value")
            });
        }
        "bool" => {
            json_object_bool_field(constant, "value", report);
        }
        _ => panic!("{label} report constant entry {index} used unknown kind {kind}"),
    }
    assert_span_schema(constant, "span", label, report);
}

fn assert_call_schema(call: &JsonValue, index: usize, label: &str, report: &str) {
    let JsonValue::Object(_) = call else {
        panic!("{label} report contained non-object call entry {index}");
    };
    let callee = json_object_string_field(call, "callee", report)
        .unwrap_or_else(|| panic!("{label} report call entry {index} contained no callee"));
    assert_nonblank(callee, || {
        format!("{label} report call entry {index} contained empty callee")
    });
    let count = json_object_u64_field(call, "count", report);
    assert!(
        count > 0,
        "{label} report call entry {index} contained zero count"
    );
    assert_span_schema(call, "span", label, report);
}

fn assert_call_site_schema(call_site: &JsonValue, index: usize, label: &str, report: &str) {
    let JsonValue::Object(_) = call_site else {
        panic!("{label} report contained non-object call_site entry {index}");
    };
    let callee = json_object_string_field(call_site, "callee", report)
        .unwrap_or_else(|| panic!("{label} report call_site entry {index} contained no callee"));
    assert_nonblank(callee, || {
        format!("{label} report call_site entry {index} contained empty callee")
    });
    assert_span_schema(call_site, "span", label, report);
    if json_object_field(call_site, "in_loop", report).is_some() {
        json_object_bool_field(call_site, "in_loop", report);
    }
    let args = json_object_array_field(call_site, "args", report);
    for (arg_index, arg) in args.iter().enumerate() {
        assert_arg_expr_schema(arg, arg_index, label, report);
    }
    if let Some(path_cond) = json_object_field(call_site, "path_cond", report) {
        assert_predicate_schema(path_cond, label, report);
    }
}

fn assert_arg_expr_schema(arg: &JsonValue, index: usize, label: &str, report: &str) {
    let JsonValue::Object(_) = arg else {
        panic!("{label} report contained non-object call arg entry {index}");
    };
    let kind = json_object_string_field(arg, "kind", report)
        .unwrap_or_else(|| panic!("{label} report call arg entry {index} contained no kind"));
    match kind {
        "param" => {
            json_object_u64_field(arg, "index", report);
        }
        "const" => {
            let value = json_object_field(arg, "value", report).unwrap_or_else(|| {
                panic!("{label} report call arg entry {index} contained no const value")
            });
            assert_constant_schema(value, index, label, report);
        }
        "nestedcall" => {
            assert_nonempty_string_field(arg, "callee", label, report);
        }
        "opaque" => {
            assert_nonempty_string_field(arg, "text", label, report);
        }
        _ => panic!("{label} report call arg entry {index} used unknown kind {kind}"),
    }
}

fn assert_predicate_schema(predicate: &JsonValue, label: &str, report: &str) {
    let JsonValue::Object(_) = predicate else {
        panic!("{label} report contained non-object path_cond");
    };
    let kind = json_object_string_field(predicate, "kind", report)
        .unwrap_or_else(|| panic!("{label} report path_cond contained no kind"));
    match kind {
        "cmp" => {
            let op = json_object_string_field(predicate, "op", report)
                .unwrap_or_else(|| panic!("{label} report path_cond cmp contained no op"));
            assert!(
                matches!(op, "lt" | "le" | "eq" | "ne" | "gt" | "ge"),
                "{label} report path_cond cmp used unknown op {op}"
            );
            for field in ["left", "right"] {
                let term = json_object_field(predicate, field, report)
                    .unwrap_or_else(|| panic!("{label} report path_cond cmp contained no {field}"));
                assert_term_schema(term, label, report);
            }
        }
        "and" | "or" => {
            let items = json_object_array_field(predicate, "items", report);
            assert!(
                !items.is_empty(),
                "{label} report path_cond {kind} contained no items"
            );
            for item in items {
                assert_predicate_schema(item, label, report);
            }
        }
        "not" => {
            let item = json_object_field(predicate, "item", report)
                .unwrap_or_else(|| panic!("{label} report path_cond not contained no item"));
            assert_predicate_schema(item, label, report);
        }
        "truthy" => {
            let term = json_object_field(predicate, "term", report)
                .unwrap_or_else(|| panic!("{label} report path_cond truthy contained no term"));
            assert_term_schema(term, label, report);
        }
        "true" | "false" => {}
        "opaque" => {
            assert_nonempty_string_field(predicate, "text", label, report);
        }
        _ => panic!("{label} report path_cond used unknown kind {kind}"),
    }
}

fn assert_term_schema(term: &JsonValue, label: &str, report: &str) {
    let JsonValue::Object(_) = term else {
        panic!("{label} report contained non-object predicate term");
    };
    let kind = json_object_string_field(term, "kind", report)
        .unwrap_or_else(|| panic!("{label} report predicate term contained no kind"));
    match kind {
        "param" => {
            json_object_u64_field(term, "index", report);
        }
        "const" => {
            let value = json_object_field(term, "value", report)
                .unwrap_or_else(|| panic!("{label} report predicate term contained no value"));
            assert_constant_schema(value, 0, label, report);
        }
        "field" => {
            let base = json_object_field(term, "base", report)
                .unwrap_or_else(|| panic!("{label} report predicate field contained no base"));
            assert_term_schema(base, label, report);
            assert_nonempty_string_field(term, "name", label, report);
        }
        "opaque" => {
            assert_nonempty_string_field(term, "text", label, report);
        }
        _ => panic!("{label} report predicate term used unknown kind {kind}"),
    }
}

fn assert_span_schema(json: &JsonValue, field: &str, label: &str, report: &str) {
    let span = json_object_array_field(json, field, report);
    assert_eq!(
        span.len(),
        2,
        "{label} report {field} did not contain exactly two entries"
    );
    let JsonValue::Number(start_text) = &span[0] else {
        panic!("{label} report {field} start was not a number");
    };
    let JsonValue::Number(end_text) = &span[1] else {
        panic!("{label} report {field} end was not a number");
    };
    let start: u64 = start_text
        .parse()
        .unwrap_or_else(|_| panic!("{label} report {field} start was not an integer"));
    let end: u64 = end_text
        .parse()
        .unwrap_or_else(|_| panic!("{label} report {field} end was not an integer"));
    assert!(
        start <= end,
        "{label} report {field} contained an inverted range"
    );
}

fn analyze(ccc: &Path, input: &str, lang: &str, output: &Path) {
    if output.exists() {
        fs::remove_file(output)
            .unwrap_or_else(|e| panic!("remove stale {} report {}: {e}", input, output.display()));
    }
    run_checked(
        Command::new(ccc)
            .current_dir(repo_root())
            .arg("analyze")
            .arg(input)
            .arg("-l")
            .arg(lang)
            .arg("--recurse")
            .arg("-o")
            .arg(output),
        &format!("ccc analyze {input}"),
    );
    assert_nonempty_file(output, &format!("ccc analyze {input}"));
    assert_analysis_report_has_functions(output, &format!("ccc analyze {input}"), lang);
}

fn assert_json_array_field_empty(stdout: &str, field: &str) {
    let json = parse_json(stdout, "ccc JSON object");
    let contents = json_object_array_field(&json, field, stdout);

    assert!(
        contents.is_empty(),
        "expected {field} to be empty, got:\n{stdout}"
    );
}

fn assert_missing_report_has_no_missing_functions(stdout: &str) {
    let json = parse_json(stdout, "ccc missing result");
    let missing = json_object_array_field(&json, "missing_in_rust", stdout);
    let extra = json_object_array_field(&json, "extra_in_rust", stdout);
    let partial = json_object_array_field(&json, "partial", stdout);

    assert_json_string_array_entries(missing, "missing_in_rust", stdout);
    assert_json_string_array_entries(extra, "extra_in_rust", stdout);
    assert_partial_entries(partial, stdout);

    assert!(
        missing.is_empty(),
        "expected missing_in_rust to be empty, got:\n{stdout}"
    );
    assert!(
        partial.is_empty(),
        "expected partial to be empty, got:\n{stdout}"
    );
}

fn assert_json_string_array_entries(entries: &[JsonValue], field: &str, stdout: &str) {
    for (index, entry) in entries.iter().enumerate() {
        let JsonValue::String(name) = entry else {
            panic!("expected {field} entry {index} to be a string, got:\n{stdout}");
        };
        assert_nonblank(name, || {
            format!("expected {field} entry {index} to be non-empty, got:\n{stdout}")
        });
    }
}

fn assert_partial_entries(entries: &[JsonValue], stdout: &str) {
    for (index, entry) in entries.iter().enumerate() {
        let JsonValue::Object(_) = entry else {
            panic!("expected partial entry {index} to be an object, got:\n{stdout}");
        };
        for field in ["rust_name", "other_name", "reason"] {
            let value = json_object_field(entry, field, stdout).unwrap_or_else(|| {
                panic!("expected partial entry {index} to contain {field}, got:\n{stdout}");
            });
            let JsonValue::String(value) = value else {
                panic!(
                    "expected partial entry {index} field {field} to be a string, got:\n{stdout}"
                );
            };
            assert_nonblank(value, || {
                format!(
                    "expected partial entry {index} field {field} to be non-empty, got:\n{stdout}"
                )
            });
        }
    }
}

fn assert_compare_json_has_mapped_pair(stdout: &str) {
    let json = parse_json(stdout, "ccc compare result");
    let JsonValue::Array(pairs) = &json else {
        panic!("expected JSON array for compare result, got:\n{stdout}");
    };
    pairs.first().unwrap_or_else(|| {
        panic!("expected ccc compare JSON to contain at least one function pair, got:\n{stdout}")
    });
    for (pair_index, pair) in pairs.iter().enumerate() {
        assert_compare_json_pair(pair, pair_index, stdout);
    }
}

fn assert_compare_json_has_exactly_one_mapped_pair(stdout: &str) {
    let json = parse_json(stdout, "ccc compare result");
    let JsonValue::Array(pairs) = &json else {
        panic!("expected JSON array for compare result, got:\n{stdout}");
    };
    assert_eq!(
        pairs.len(),
        1,
        "expected ccc compare --top 1 JSON to contain exactly one function pair, got:\n{stdout}"
    );
    assert_compare_json_pair(&pairs[0], 0, stdout);
}

fn assert_missing_structs_report_has_no_missing_structs(stdout: &str) {
    let json = parse_json(stdout, "ccc missing-structs result");
    let missing = json_object_array_field(&json, "missing_in_rust", stdout);
    let extra = json_object_array_field(&json, "extra_in_rust", stdout);

    assert_json_string_array_entries(missing, "missing_in_rust", stdout);
    assert_json_string_array_entries(extra, "extra_in_rust", stdout);

    assert!(
        missing.is_empty(),
        "expected missing_in_rust structs to be empty, got:\n{stdout}"
    );
}

fn assert_compare_structs_json_has_exactly_one_mapped_pair(stdout: &str) {
    let json = parse_json(stdout, "ccc compare-structs result");
    let JsonValue::Array(pairs) = &json else {
        panic!("expected JSON array for compare-structs result, got:\n{stdout}");
    };
    assert_eq!(
        pairs.len(),
        1,
        "expected ccc compare-structs --top 1 JSON to contain exactly one struct pair, got:\n{stdout}"
    );
    assert_compare_structs_json_pair(&pairs[0], 0, stdout);
}

fn assert_compare_structs_json_pair(pair: &JsonValue, pair_index: usize, stdout: &str) {
    let JsonValue::Object(_) = pair else {
        panic!(
            "expected compare-structs result element {pair_index} to be an object, got:\n{stdout}"
        );
    };

    for field in ["rust_name", "other_name"] {
        let value = json_object_field(pair, field, stdout).unwrap_or_else(|| {
            panic!(
                "expected ccc compare-structs result {pair_index} to contain {field}, got:\n{stdout}"
            );
        });
        let JsonValue::String(name) = value else {
            panic!(
                "expected ccc compare-structs result {pair_index} field {field} to be a string, got:\n{stdout}"
            );
        };
        assert_nonblank(name, || {
            format!(
                "expected ccc compare-structs result {pair_index} field {field} to be non-empty, got:\n{stdout}"
            )
        });
    }

    let total = json_object_field(pair, "total", stdout).unwrap_or_else(|| {
        panic!("expected ccc compare-structs result {pair_index} to contain total, got:\n{stdout}");
    });
    let JsonValue::Number(_) = total else {
        panic!(
            "expected ccc compare-structs result {pair_index} field total to be a number, got:\n{stdout}"
        );
    };
    let total = nonnegative_json_number(total, stdout);

    let per_category = json_object_field(pair, "per_category", stdout).unwrap_or_else(|| {
        panic!(
            "expected ccc compare-structs result {pair_index} to contain per_category, got:\n{stdout}"
        );
    });
    let JsonValue::Array(per_category) = per_category else {
        panic!(
            "expected ccc compare-structs result {pair_index} field per_category to be an array, got:\n{stdout}"
        );
    };
    assert!(
        !per_category.is_empty(),
        "expected ccc compare-structs result {pair_index} field per_category to be non-empty, got:\n{stdout}"
    );
    let mut seen_categories = Vec::new();
    let mut contribution_total = 0.0;
    for (category_index, category) in per_category.iter().enumerate() {
        let (category_name, contribution) =
            assert_compare_struct_category_tuple(category, pair_index, category_index, stdout);
        assert!(
            !seen_categories.iter().any(|seen| seen == category_name),
            "expected ccc compare-structs result {pair_index} category {category_index} to be unique, got:\n{stdout}"
        );
        seen_categories.push(category_name.to_string());
        contribution_total += contribution;
    }
    assert_json_total_matches_contributions(total, contribution_total, "compare-structs", stdout);
}

fn assert_compare_struct_category_tuple<'a>(
    category: &'a JsonValue,
    pair_index: usize,
    category_index: usize,
    stdout: &str,
) -> (&'a str, f64) {
    let JsonValue::Array(category_fields) = category else {
        panic!(
            "expected ccc compare-structs result {pair_index} category {category_index} to be an array, got:\n{stdout}"
        );
    };
    assert_eq!(
        category_fields.len(),
        4,
        "expected ccc compare-structs result {pair_index} category {category_index} to have 4 fields, got:\n{stdout}"
    );
    let Some(JsonValue::String(category_name)) = category_fields.first() else {
        panic!("expected ccc compare-structs result {pair_index} category {category_index} name to be a string, got:\n{stdout}");
    };
    assert_nonblank(category_name, || {
        format!(
            "expected ccc compare-structs result {pair_index} category {category_index} name to be non-empty, got:\n{stdout}"
        )
    });
    assert!(
        matches!(
            category_name.as_str(),
            "field_count"
                | "int"
                | "float"
                | "bool"
                | "char"
                | "string"
                | "pointer"
                | "array"
                | "collection"
                | "other"
        ),
        "expected ccc compare-structs result {pair_index} category {category_index} to use a known category, got:\n{stdout}"
    );
    let mut contribution = 0.0;
    for (field_index, value) in category_fields.iter().enumerate().skip(1) {
        let JsonValue::Number(_) = value else {
            panic!(
                "expected ccc compare-structs result {pair_index} category {category_index} field {field_index} to be a number, got:\n{stdout}"
            );
        };
        let value = nonnegative_json_number(value, stdout);
        if field_index == 3 {
            contribution = value;
        }
    }
    (category_name.as_str(), contribution)
}

fn assert_compare_json_pair(pair: &JsonValue, pair_index: usize, stdout: &str) {
    let JsonValue::Object(_) = pair else {
        panic!("expected compare result element {pair_index} to be an object, got:\n{stdout}");
    };

    for field in ["rust_name", "other_name"] {
        let value = json_object_field(pair, field, stdout).unwrap_or_else(|| {
            panic!("expected ccc compare result {pair_index} to contain {field}, got:\n{stdout}");
        });
        let JsonValue::String(name) = value else {
            panic!(
                "expected ccc compare result {pair_index} field {field} to be a string, got:\n{stdout}"
            );
        };
        assert_nonblank(name, || {
            format!(
                "expected ccc compare result {pair_index} field {field} to be non-empty, got:\n{stdout}"
            )
        });
    }

    let per_metric = json_object_field(pair, "per_metric", stdout).unwrap_or_else(|| {
        panic!("expected ccc compare result {pair_index} to contain per_metric, got:\n{stdout}");
    });
    let total = json_object_field(pair, "total", stdout).unwrap_or_else(|| {
        panic!("expected ccc compare result {pair_index} to contain total, got:\n{stdout}");
    });
    let JsonValue::Number(_) = total else {
        panic!(
            "expected ccc compare result {pair_index} field total to be a number, got:\n{stdout}"
        );
    };
    let total = nonnegative_json_number(total, stdout);
    let JsonValue::Array(per_metric) = per_metric else {
        panic!("expected ccc compare result {pair_index} field per_metric to be an array, got:\n{stdout}");
    };
    assert!(
        !per_metric.is_empty(),
        "expected ccc compare result {pair_index} field per_metric to be non-empty, got:\n{stdout}"
    );
    let mut seen_metrics = Vec::new();
    let mut contribution_total = 0.0;
    for (metric_index, metric) in per_metric.iter().enumerate() {
        let (metric_name, contribution) =
            assert_compare_metric_tuple(metric, pair_index, metric_index, stdout);
        assert!(
            !seen_metrics.iter().any(|seen| seen == metric_name),
            "expected ccc compare result {pair_index} metric {metric_index} to be unique, got:\n{stdout}"
        );
        seen_metrics.push(metric_name.to_string());
        contribution_total += contribution;
    }
    assert_json_total_matches_contributions(total, contribution_total, "compare", stdout);
}

fn assert_compare_metric_tuple<'a>(
    metric: &'a JsonValue,
    pair_index: usize,
    metric_index: usize,
    stdout: &str,
) -> (&'a str, f64) {
    let JsonValue::Array(metric_fields) = metric else {
        panic!(
            "expected ccc compare result {pair_index} metric {metric_index} to be an array, got:\n{stdout}"
        );
    };
    assert_eq!(
        metric_fields.len(),
        4,
        "expected ccc compare result {pair_index} metric {metric_index} to have 4 fields, got:\n{stdout}"
    );
    let Some(JsonValue::String(metric_name)) = metric_fields.first() else {
        panic!("expected ccc compare result {pair_index} metric {metric_index} name to be a string, got:\n{stdout}");
    };
    assert_nonblank(metric_name, || {
        format!(
            "expected ccc compare result {pair_index} metric {metric_index} name to be non-empty, got:\n{stdout}"
        )
    });
    assert!(
        is_known_compare_metric(metric_name),
        "expected ccc compare result {pair_index} metric {metric_index} to use a known metric, got:\n{stdout}"
    );
    let mut contribution = 0.0;
    for (field_index, value) in metric_fields.iter().enumerate().skip(1) {
        let JsonValue::Number(_) = value else {
            panic!(
                "expected ccc compare result {pair_index} metric {metric_index} field {field_index} to be a number, got:\n{stdout}"
            );
        };
        let value = nonnegative_json_number(value, stdout);
        if field_index == 3 {
            contribution = value;
        }
    }
    (metric_name.as_str(), contribution)
}

fn assert_json_total_matches_contributions(
    total: f64,
    contribution_total: f64,
    label: &str,
    stdout: &str,
) {
    let tolerance = 1e-9 * total.max(contribution_total).max(1.0);
    assert!(
        (total - contribution_total).abs() <= tolerance,
        "expected ccc {label} total to equal summed contributions, got:\n{stdout}"
    );
}

fn is_known_compare_metric(metric_name: &str) -> bool {
    matches!(
        metric_name,
        "loc_code"
            | "loc_comments"
            | "loc_asm"
            | "inputs"
            | "outputs"
            | "branches"
            | "loops"
            | "max_loop_nesting"
            | "max_if_nesting"
            | "max_combined_nesting"
            | "calls_unique"
            | "calls_total"
            | "cyclomatic"
            | "cognitive"
            | "early_returns"
            | "goto_count"
            | "unsafe_blocks"
            | "halstead_n1"
            | "halstead_n2"
            | "halstead_big_n1"
            | "halstead_big_n2"
            | "halstead_volume"
            | "halstead_difficulty"
            | "op_add"
            | "op_sub"
            | "op_mul"
            | "op_div"
            | "op_rem"
            | "op_shl"
            | "op_shr"
            | "op_bit_and"
            | "op_bit_or"
            | "op_bit_xor"
            | "op_bit_not"
            | "op_logic_and"
            | "op_logic_or"
            | "op_logic_not"
    )
}

#[derive(Debug)]
enum JsonValue {
    Object(Vec<(String, JsonValue)>),
    Array(Vec<JsonValue>),
    String(String),
    Number(String),
    Bool(bool),
    Null,
}

fn parse_json(input: &str, label: &str) -> JsonValue {
    let mut parser = JsonParser { input, cursor: 0 };
    let value = parser.parse_value(label);
    parser.skip_whitespace();
    assert!(
        parser.is_eof(),
        "expected complete JSON {label}, got trailing data:\n{input}"
    );
    value
}

fn json_object_array_field<'a>(json: &'a JsonValue, field: &str, stdout: &str) -> &'a [JsonValue] {
    let value = json_object_field(json, field, stdout)
        .unwrap_or_else(|| panic!("missing top-level JSON field {field} in:\n{stdout}"));
    let JsonValue::Array(array) = value else {
        panic!("expected JSON array for {field}, got:\n{stdout}");
    };
    array
}

fn json_object_array_field_optional<'a>(
    json: &'a JsonValue,
    field: &str,
    stdout: &str,
) -> Option<&'a [JsonValue]> {
    json_object_field(json, field, stdout).map(|value| {
        let JsonValue::Array(array) = value else {
            panic!("expected JSON array for {field}, got:\n{stdout}");
        };
        array.as_slice()
    })
}

fn json_object_field<'a>(json: &'a JsonValue, field: &str, stdout: &str) -> Option<&'a JsonValue> {
    let JsonValue::Object(fields) = json else {
        panic!("expected JSON object while looking for {field}, got:\n{stdout}");
    };
    let mut matches = fields
        .iter()
        .filter_map(|(key, value)| (key == field).then_some(value));
    let value = matches.next();
    assert!(
        matches.next().is_none(),
        "duplicate JSON field {field} in:\n{stdout}"
    );
    value
}

fn json_object_string_field<'a>(json: &'a JsonValue, field: &str, stdout: &str) -> Option<&'a str> {
    json_object_field(json, field, stdout).map(|value| {
        let JsonValue::String(value) = value else {
            panic!("expected JSON string for {field}, got:\n{stdout}");
        };
        value.as_str()
    })
}

fn assert_nonempty_string_field(json: &JsonValue, field: &str, label: &str, stdout: &str) {
    let value = json_object_string_field(json, field, stdout)
        .unwrap_or_else(|| panic!("{label} report contained no {field}"));
    assert_nonblank(value, || format!("{label} report contained empty {field}"));
}

fn assert_optional_nonblank_string_field(json: &JsonValue, field: &str, label: &str, stdout: &str) {
    if let Some(value) = json_object_string_field(json, field, stdout) {
        assert_nonblank(value, || format!("{label} report contained empty {field}"));
    }
}

fn assert_attributes_schema(json: &JsonValue, label: &str, stdout: &str) {
    let Some(attributes) = json_object_field(json, "attributes", stdout) else {
        return;
    };
    let JsonValue::Object(fields) = attributes else {
        panic!("{label} report contained non-object attributes");
    };
    for (key, value) in fields {
        assert_nonblank(key, || {
            format!("{label} report contained empty attribute key")
        });
        let JsonValue::String(value) = value else {
            panic!("{label} report contained non-string attribute value for {key}");
        };
        assert_nonblank(value, || {
            format!("{label} report contained empty attribute value for {key}")
        });
    }
}

fn assert_nonblank<F>(value: &str, message: F)
where
    F: FnOnce() -> String,
{
    assert!(!value.trim().is_empty(), "{}", message());
}

fn assert_nonempty<F>(value: &str, message: F)
where
    F: FnOnce() -> String,
{
    assert!(!value.is_empty(), "{}", message());
}

fn assert_empty_or_nonblank<F>(value: &str, message: F)
where
    F: FnOnce() -> String,
{
    assert!(
        value.is_empty() || !value.trim().is_empty(),
        "{}",
        message()
    );
}

fn json_object_u64_field(json: &JsonValue, field: &str, stdout: &str) -> u64 {
    let value = json_object_field(json, field, stdout)
        .unwrap_or_else(|| panic!("missing JSON field {field} in:\n{stdout}"));
    let JsonValue::Number(value) = value else {
        panic!("expected JSON integer for {field}, got:\n{stdout}");
    };
    value
        .parse()
        .unwrap_or_else(|_| panic!("expected JSON integer for {field}, got:\n{stdout}"))
}

fn json_object_i64_field(json: &JsonValue, field: &str, stdout: &str) -> i64 {
    let value = json_object_field(json, field, stdout)
        .unwrap_or_else(|| panic!("missing JSON field {field} in:\n{stdout}"));
    let JsonValue::Number(value) = value else {
        panic!("expected JSON integer for {field}, got:\n{stdout}");
    };
    value
        .parse()
        .unwrap_or_else(|_| panic!("expected JSON integer for {field}, got:\n{stdout}"))
}

fn json_object_bool_field(json: &JsonValue, field: &str, stdout: &str) -> bool {
    let value = json_object_field(json, field, stdout)
        .unwrap_or_else(|| panic!("missing JSON field {field} in:\n{stdout}"));
    let JsonValue::Bool(value) = value else {
        panic!("expected JSON bool for {field}, got:\n{stdout}");
    };
    *value
}

fn finite_json_number_field(json: &JsonValue, field: &str, stdout: &str) -> f64 {
    let value = json_object_field(json, field, stdout)
        .unwrap_or_else(|| panic!("missing JSON field {field} in:\n{stdout}"));
    finite_json_number(value, stdout)
}

fn nonnegative_json_number_field(json: &JsonValue, field: &str, stdout: &str) -> f64 {
    let value = json_object_field(json, field, stdout)
        .unwrap_or_else(|| panic!("missing JSON field {field} in:\n{stdout}"));
    nonnegative_json_number(value, stdout)
}

fn nonnegative_json_number(json: &JsonValue, stdout: &str) -> f64 {
    let number = finite_json_number(json, stdout);
    assert!(
        number >= 0.0,
        "expected nonnegative JSON number, got:\n{stdout}"
    );
    number
}

fn finite_json_number(json: &JsonValue, stdout: &str) -> f64 {
    let JsonValue::Number(value) = json else {
        panic!("expected finite JSON number, got:\n{stdout}");
    };
    let number: f64 = value
        .parse()
        .unwrap_or_else(|_| panic!("expected finite JSON number, got:\n{stdout}"));
    assert!(
        number.is_finite(),
        "expected finite JSON number, got:\n{stdout}"
    );
    number
}

fn count_finite_numbers(json: &JsonValue, stdout: &str) -> usize {
    match json {
        JsonValue::Number(value) => {
            let value = JsonValue::Number(value.clone());
            finite_json_number(&value, stdout);
            1
        }
        JsonValue::Object(fields) => fields
            .iter()
            .map(|(_, value)| count_finite_numbers(value, stdout))
            .sum(),
        JsonValue::Array(values) => values
            .iter()
            .map(|value| count_finite_numbers(value, stdout))
            .sum(),
        JsonValue::String(_) | JsonValue::Bool(_) | JsonValue::Null => 0,
    }
}

struct JsonParser<'a> {
    input: &'a str,
    cursor: usize,
}

impl JsonParser<'_> {
    fn is_eof(&self) -> bool {
        self.cursor == self.input.len()
    }

    fn rest(&self) -> &str {
        &self.input[self.cursor..]
    }

    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.cursor += ch.len_utf8();
        Some(ch)
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(' ' | '\n' | '\r' | '\t')) {
            self.bump();
        }
    }

    fn parse_value(&mut self, label: &str) -> JsonValue {
        self.skip_whitespace();
        match self.peek() {
            Some('{') => self.parse_object(label),
            Some('[') => self.parse_array(label),
            Some('"') => JsonValue::String(self.parse_string(label)),
            Some('t') => {
                self.expect_literal("true", label);
                JsonValue::Bool(true)
            }
            Some('f') => {
                self.expect_literal("false", label);
                JsonValue::Bool(false)
            }
            Some('n') => {
                self.expect_literal("null", label);
                JsonValue::Null
            }
            Some('-' | '0'..='9') => JsonValue::Number(self.parse_number(label)),
            _ => panic!("expected JSON value for {label}, got:\n{}", self.input),
        }
    }

    fn parse_object(&mut self, label: &str) -> JsonValue {
        self.expect_char('{', label);
        let mut fields = Vec::new();
        self.skip_whitespace();
        if self.consume_char('}') {
            return JsonValue::Object(fields);
        }

        loop {
            self.skip_whitespace();
            if self.peek() != Some('"') {
                panic!("expected JSON object key for {label}, got:\n{}", self.input);
            }
            let key = self.parse_string(label);
            self.skip_whitespace();
            self.expect_char(':', label);
            let value = self.parse_value(label);
            assert!(
                !fields.iter().any(|(existing_key, _)| existing_key == &key),
                "duplicate JSON field {key} for {label}, got:\n{}",
                self.input
            );
            fields.push((key, value));
            self.skip_whitespace();
            if self.consume_char('}') {
                break;
            }
            self.expect_char(',', label);
        }

        JsonValue::Object(fields)
    }

    fn parse_array(&mut self, label: &str) -> JsonValue {
        self.expect_char('[', label);
        let mut values = Vec::new();
        self.skip_whitespace();
        if self.consume_char(']') {
            return JsonValue::Array(values);
        }

        loop {
            values.push(self.parse_value(label));
            self.skip_whitespace();
            if self.consume_char(']') {
                break;
            }
            self.expect_char(',', label);
        }

        JsonValue::Array(values)
    }

    fn parse_string(&mut self, label: &str) -> String {
        self.expect_char('"', label);
        let mut value = String::new();
        loop {
            let ch = self
                .bump()
                .unwrap_or_else(|| panic!("unterminated JSON string for {label}"));
            match ch {
                '"' => return value,
                '\\' => value.push(self.parse_escape(label)),
                ch if ch <= '\u{1f}' => {
                    panic!("control character in JSON string for {label}: {ch:?}")
                }
                ch => value.push(ch),
            }
        }
    }

    fn parse_escape(&mut self, label: &str) -> char {
        match self
            .bump()
            .unwrap_or_else(|| panic!("unterminated JSON escape for {label}"))
        {
            '"' => '"',
            '\\' => '\\',
            '/' => '/',
            'b' => '\u{8}',
            'f' => '\u{c}',
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            'u' => {
                let code = self.parse_hex_escape(label);
                if (0xd800..=0xdbff).contains(&code) {
                    if self.bump() != Some('\\') || self.bump() != Some('u') {
                        panic!("invalid JSON unicode surrogate pair for {label}");
                    }
                    let low = self.parse_hex_escape(label);
                    assert!(
                        (0xdc00..=0xdfff).contains(&low),
                        "invalid JSON unicode surrogate pair for {label}"
                    );
                    let scalar = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
                    char::from_u32(scalar)
                        .unwrap_or_else(|| panic!("invalid JSON unicode scalar for {label}"))
                } else {
                    assert!(
                        !(0xdc00..=0xdfff).contains(&code),
                        "invalid JSON unicode surrogate pair for {label}"
                    );
                    char::from_u32(code)
                        .unwrap_or_else(|| panic!("invalid JSON unicode scalar for {label}"))
                }
            }
            ch => panic!("invalid JSON escape \\{ch} for {label}"),
        }
    }

    fn parse_hex_escape(&mut self, label: &str) -> u32 {
        let mut code = 0u32;
        for _ in 0..4 {
            let ch = self
                .bump()
                .unwrap_or_else(|| panic!("invalid JSON unicode escape for {label}"));
            let digit = ch
                .to_digit(16)
                .unwrap_or_else(|| panic!("invalid JSON unicode escape for {label}"));
            code = (code << 4) | digit;
        }
        code
    }

    fn parse_number(&mut self, label: &str) -> String {
        let start = self.cursor;
        self.consume_char('-');
        match self.peek() {
            Some('0') => {
                self.bump();
            }
            Some('1'..='9') => {
                self.bump();
                while matches!(self.peek(), Some('0'..='9')) {
                    self.bump();
                }
            }
            _ => panic!("invalid JSON number for {label}, got:\n{}", self.input),
        }
        if self.consume_char('.') {
            let mut digits = 0usize;
            while matches!(self.peek(), Some('0'..='9')) {
                self.bump();
                digits += 1;
            }
            assert!(
                digits > 0,
                "invalid JSON number fraction for {label}, got:\n{}",
                self.input
            );
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            self.bump();
            if matches!(self.peek(), Some('+' | '-')) {
                self.bump();
            }
            let mut digits = 0usize;
            while matches!(self.peek(), Some('0'..='9')) {
                self.bump();
                digits += 1;
            }
            assert!(
                digits > 0,
                "invalid JSON number exponent for {label}, got:\n{}",
                self.input
            );
        }
        self.input[start..self.cursor].to_string()
    }

    fn expect_literal(&mut self, literal: &str, label: &str) {
        if !self.rest().starts_with(literal) {
            panic!(
                "expected JSON literal {literal} for {label}, got:\n{}",
                self.input
            );
        }
        self.cursor += literal.len();
    }

    fn expect_char(&mut self, expected: char, label: &str) {
        let got = self
            .bump()
            .unwrap_or_else(|| panic!("expected {expected:?} for {label}, got end of input"));
        assert_eq!(
            got, expected,
            "expected {expected:?} for {label}, got {got:?} in:\n{}",
            self.input
        );
    }

    fn consume_char(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.bump();
            true
        } else {
            false
        }
    }
}

fn analysis_report_json(functions: &str) -> String {
    format!(
        r#"{{"schema_version":4,"language":"rust","source_file":"src/decompress","source_hash":"hash","functions":{functions}}}"#
    )
}

fn analysis_function_json(name: &str) -> String {
    format!(
        r#"{{"name":"{name}","location":{{"file":"src/decompress/mod.rs","line_start":1,"line_end":2,"col_start":0,"col_end":1,"byte_start":0,"byte_end":10}},"signature":{{"inputs":[],"outputs":[]}},"metrics":{{"loc_code":1,"loc_comments":0,"loc_asm":0,"inputs":0,"outputs":0,"branches":0,"loops":0,"max_loop_nesting":0,"max_if_nesting":0,"max_combined_nesting":0,"calls_unique":0,"calls_total":0,"cyclomatic":1,"cognitive":0,"halstead":{{"n1":0,"n2":0,"big_n1":0,"big_n2":0,"volume":0.0,"difficulty":0.0}},"early_returns":0,"goto_count":0,"unsafe_blocks":0,"binary_operators":{{"add":0,"sub":0,"mul":0,"div":0,"rem":0,"shl":0,"shr":0,"bit_and":0,"bit_or":0,"bit_xor":0,"bit_not":0,"logic_and":0,"logic_or":0,"logic_not":0}}}},"constants":[],"calls":[],"types_used":[]}}"#
    )
}

#[test]
fn json_array_field_contents_reads_only_top_level_fields() {
    let stdout = r#"{
        "extra_in_rust": [{"partial": ["nested"]}],
        "partial": [
            {"rust_name": "stub", "reason": "uses ] and [ inside a string"}
        ],
        "missing_in_rust": []
    }"#;

    let json = parse_json(stdout, "test JSON object");
    assert!(json_object_array_field(&json, "missing_in_rust", stdout).is_empty());
    assert!(!json_object_array_field(&json, "partial", stdout).is_empty());
}

#[test]
fn missing_report_assertion_allows_extra_rust_functions() {
    assert_missing_report_has_no_missing_functions(
        r#"{"missing_in_rust":[],"extra_in_rust":["helper"],"partial":[]}"#,
    );
}

#[test]
#[should_panic(expected = "expected missing_in_rust entry 0 to be a string")]
fn missing_report_assertion_rejects_non_string_missing_entry() {
    assert_missing_report_has_no_missing_functions(
        r#"{"missing_in_rust":[{"name":"missing"}],"extra_in_rust":[],"partial":[]}"#,
    );
}

#[test]
#[should_panic(expected = "expected extra_in_rust entry 0 to be a string")]
fn missing_report_assertion_rejects_non_string_extra_entry() {
    assert_missing_report_has_no_missing_functions(
        r#"{"missing_in_rust":[],"extra_in_rust":[1],"partial":[]}"#,
    );
}

#[test]
#[should_panic(expected = "expected extra_in_rust entry 0 to be non-empty")]
fn missing_report_assertion_rejects_empty_extra_entry() {
    assert_missing_report_has_no_missing_functions(
        r#"{"missing_in_rust":[],"extra_in_rust":[""],"partial":[]}"#,
    );
}

#[test]
#[should_panic(expected = "expected extra_in_rust entry 0 to be non-empty")]
fn missing_report_assertion_rejects_blank_extra_entry() {
    assert_missing_report_has_no_missing_functions(
        r#"{"missing_in_rust":[],"extra_in_rust":["   "],"partial":[]}"#,
    );
}

#[test]
#[should_panic(expected = "expected partial entry 0 to be an object")]
fn missing_report_assertion_rejects_non_object_partial_entry() {
    assert_missing_report_has_no_missing_functions(
        r#"{"missing_in_rust":[],"extra_in_rust":[],"partial":["stub"]}"#,
    );
}

#[test]
#[should_panic(expected = "expected partial entry 0 to contain reason")]
fn missing_report_assertion_rejects_partial_entry_missing_reason() {
    assert_missing_report_has_no_missing_functions(
        r#"{"missing_in_rust":[],"extra_in_rust":[],"partial":[{"rust_name":"r","other_name":"c"}]}"#,
    );
}

#[test]
#[should_panic(expected = "expected partial entry 0 field rust_name to be a string")]
fn missing_report_assertion_rejects_non_string_partial_name() {
    assert_missing_report_has_no_missing_functions(
        r#"{"missing_in_rust":[],"extra_in_rust":[],"partial":[{"rust_name":1,"other_name":"c","reason":"stub"}]}"#,
    );
}

#[test]
#[should_panic(expected = "expected partial entry 0 field reason to be non-empty")]
fn missing_report_assertion_rejects_empty_partial_reason() {
    assert_missing_report_has_no_missing_functions(
        r#"{"missing_in_rust":[],"extra_in_rust":[],"partial":[{"rust_name":"r","other_name":"c","reason":""}]}"#,
    );
}

#[test]
#[should_panic(expected = "expected partial to be empty")]
fn missing_report_assertion_rejects_well_formed_partial_entries() {
    assert_missing_report_has_no_missing_functions(
        r#"{"missing_in_rust":[],"extra_in_rust":[],"partial":[{"rust_name":"r","other_name":"c","reason":"rust LOC 1 is 1% of other LOC 100"}]}"#,
    );
}

#[test]
#[should_panic(expected = "missing top-level JSON field extra_in_rust")]
fn missing_report_assertion_requires_extra_rust_field() {
    assert_missing_report_has_no_missing_functions(r#"{"missing_in_rust":[],"partial":[]}"#);
}

#[test]
fn analysis_report_assertion_requires_named_functions() {
    assert_analysis_report_contents_has_functions(
        &analysis_report_json(&format!(
            "[{}]",
            analysis_function_json("ZSTD_decompressMultiFrame")
        )),
        "test analyze",
        Some("rust"),
    );
}

#[test]
fn analysis_report_assertion_accepts_escaped_json_field_names() {
    let report = analysis_report_json(&format!(
        "[{}]",
        analysis_function_json("ZSTD_decompressMultiFrame")
    ))
    .replace("\"functions\"", "\"funct\\u0069ons\"")
    .replace("\"name\"", "\"n\\u0061me\"")
    .replace("\"location\"", "\"loc\\u0061tion\"");
    assert_analysis_report_contents_has_functions(&report, "test analyze", Some("rust"));
}

#[test]
#[should_panic(expected = "non-object function entry")]
fn analysis_report_assertion_rejects_non_object_function() {
    assert_analysis_report_contents_has_functions(
        &analysis_report_json("[null]"),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "function without a name")]
fn analysis_report_assertion_rejects_missing_function_name() {
    assert_analysis_report_contents_has_functions(
        &analysis_report_json("[{}]"),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "expected JSON string for name")]
fn analysis_report_assertion_rejects_non_string_function_name() {
    assert_analysis_report_contents_has_functions(
        &analysis_report_json(r#"[{"name":1}]"#),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "function with an empty name")]
fn analysis_report_assertion_rejects_empty_function_name() {
    assert_analysis_report_contents_has_functions(
        &analysis_report_json(r#"[{"name":""}]"#),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "function with an empty name")]
fn analysis_report_assertion_rejects_blank_function_name() {
    assert_analysis_report_contents_has_functions(
        &analysis_report_json(r#"[{"name":"   "}]"#),
        "test analyze",
        Some("rust"),
    );
}

#[test]
fn analysis_report_assertion_allows_empty_source_hash() {
    assert_analysis_report_contents_has_functions(
        &format!(
            r#"{{"schema_version":4,"language":"rust","source_file":"src/decompress","source_hash":"","functions":[{}]}}"#,
            analysis_function_json("ZSTD_decompressMultiFrame")
        ),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "contained a blank source_hash")]
fn analysis_report_assertion_rejects_blank_source_hash() {
    assert_analysis_report_contents_has_functions(
        &format!(
            r#"{{"schema_version":4,"language":"rust","source_file":"src/decompress","source_hash":"   ","functions":[{}]}}"#,
            analysis_function_json("ZSTD_decompressMultiFrame")
        ),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "contained empty original_name")]
fn analysis_report_assertion_rejects_blank_optional_function_metadata() {
    let function = analysis_function_json("ZSTD_decompressMultiFrame").replace(
        r#""name":"ZSTD_decompressMultiFrame","#,
        r#""name":"ZSTD_decompressMultiFrame","original_name":"   ","#,
    );

    assert_analysis_report_contents_has_functions(
        &analysis_report_json(&format!("[{function}]")),
        "test analyze",
        Some("rust"),
    );
}

#[test]
fn analysis_report_assertion_accepts_well_formed_structs() {
    assert_analysis_report_contents_has_functions(
        &format!(
            r#"{{"schema_version":4,"language":"rust","source_file":"src/decompress","source_hash":"hash","functions":[{}],"structs":[{{"name":"ZSTD_DCtx","kind":"struct","location":{{"file":"src/decompress/mod.rs","line_start":1,"line_end":2,"col_start":0,"col_end":1,"byte_start":0,"byte_end":10}},"fields":[{{"name":"stage","ty":{{"text":"u32"}},"category":"int"}},{{"name":"workspace","ty":{{"text":"*mut u8"}},"category":"pointer"}}],"metrics":{{"field_count":2,"int_count":1,"float_count":0,"bool_count":0,"char_count":0,"string_count":0,"pointer_count":1,"array_count":0,"collection_count":0,"other_count":0}}}}]}}"#,
            analysis_function_json("ZSTD_decompressMultiFrame")
        ),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "field_count did not match fields")]
fn analysis_report_assertion_rejects_inconsistent_struct_metrics() {
    assert_analysis_report_contents_has_functions(
        &format!(
            r#"{{"schema_version":4,"language":"rust","source_file":"src/decompress","source_hash":"hash","functions":[{}],"structs":[{{"name":"ZSTD_DCtx","kind":"struct","location":{{"file":"src/decompress/mod.rs","line_start":1,"line_end":2,"col_start":0,"col_end":1,"byte_start":0,"byte_end":10}},"fields":[{{"name":"stage","ty":{{"text":"u32"}},"category":"int"}}],"metrics":{{"field_count":2,"int_count":1,"float_count":0,"bool_count":0,"char_count":0,"string_count":0,"pointer_count":0,"array_count":0,"collection_count":0,"other_count":0}}}}]}}"#,
            analysis_function_json("ZSTD_decompressMultiFrame")
        ),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "int_count did not match fields")]
fn analysis_report_assertion_rejects_struct_category_count_mismatch() {
    assert_analysis_report_contents_has_functions(
        &format!(
            r#"{{"schema_version":4,"language":"rust","source_file":"src/decompress","source_hash":"hash","functions":[{}],"structs":[{{"name":"ZSTD_DCtx","kind":"struct","location":{{"file":"src/decompress/mod.rs","line_start":1,"line_end":2,"col_start":0,"col_end":1,"byte_start":0,"byte_end":10}},"fields":[{{"name":"workspace","ty":{{"text":"*mut u8"}},"category":"pointer"}}],"metrics":{{"field_count":1,"int_count":1,"float_count":0,"bool_count":0,"char_count":0,"string_count":0,"pointer_count":0,"array_count":0,"collection_count":0,"other_count":0}}}}]}}"#,
            analysis_function_json("ZSTD_decompressMultiFrame")
        ),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "used unknown category")]
fn analysis_report_assertion_rejects_unknown_struct_field_category() {
    assert_analysis_report_contents_has_functions(
        &format!(
            r#"{{"schema_version":4,"language":"rust","source_file":"src/decompress","source_hash":"hash","functions":[{}],"structs":[{{"name":"ZSTD_DCtx","kind":"struct","location":{{"file":"src/decompress/mod.rs","line_start":1,"line_end":2,"col_start":0,"col_end":1,"byte_start":0,"byte_end":10}},"fields":[{{"name":"stage","ty":{{"text":"u32"}},"category":"number"}}],"metrics":{{"field_count":1,"int_count":1,"float_count":0,"bool_count":0,"char_count":0,"string_count":0,"pointer_count":0,"array_count":0,"collection_count":0,"other_count":0}}}}]}}"#,
            analysis_function_json("ZSTD_decompressMultiFrame")
        ),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "contained no location")]
fn analysis_report_assertion_rejects_missing_location() {
    assert_analysis_report_contents_has_functions(
        &analysis_report_json(r#"[{"name":"ok","metrics":{"loc_code":1}}]"#),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "inverted line span")]
fn analysis_report_assertion_rejects_inverted_location_span() {
    assert_analysis_report_contents_has_functions(
        &analysis_report_json(
            r#"[{"name":"ok","location":{"file":"src/decompress/mod.rs","line_start":3,"line_end":2,"col_start":0,"col_end":1,"byte_start":0,"byte_end":10},"metrics":{"loc_code":1}}]"#,
        ),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "contained no metrics")]
fn analysis_report_assertion_rejects_missing_metrics() {
    assert_analysis_report_contents_has_functions(
        &analysis_report_json(
            r#"[{"name":"ok","location":{"file":"src/decompress/mod.rs","line_start":1,"line_end":2,"col_start":0,"col_end":1,"byte_start":0,"byte_end":10}}]"#,
        ),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "contained no numeric metrics")]
fn analysis_report_assertion_rejects_metrics_without_numbers() {
    assert_analysis_report_contents_has_functions(
        &analysis_report_json(
            r#"[{"name":"ok","location":{"file":"src/decompress/mod.rs","line_start":1,"line_end":2,"col_start":0,"col_end":1,"byte_start":0,"byte_end":10},"metrics":{"kind":"stub"}}]"#,
        ),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "contained no signature")]
fn analysis_report_assertion_rejects_missing_signature() {
    assert_analysis_report_contents_has_functions(
        &analysis_report_json(
            r#"[{"name":"ok","location":{"file":"src/decompress/mod.rs","line_start":1,"line_end":2,"col_start":0,"col_end":1,"byte_start":0,"byte_end":10},"metrics":{"loc_code":1,"loc_comments":0,"loc_asm":0,"inputs":0,"outputs":0,"branches":0,"loops":0,"max_loop_nesting":0,"max_if_nesting":0,"max_combined_nesting":0,"calls_unique":0,"calls_total":0,"cyclomatic":1,"cognitive":0,"halstead":{"n1":0,"n2":0,"big_n1":0,"big_n2":0,"volume":0,"difficulty":0},"early_returns":0,"goto_count":0,"unsafe_blocks":0,"binary_operators":{"add":0,"sub":0,"mul":0,"div":0,"rem":0,"shl":0,"shr":0,"bit_and":0,"bit_or":0,"bit_xor":0,"bit_not":0,"logic_and":0,"logic_or":0,"logic_not":0}},"constants":[],"calls":[],"types_used":[]}]"#,
        ),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "contained no halstead metrics")]
fn analysis_report_assertion_rejects_missing_halstead_metrics() {
    assert_analysis_report_contents_has_functions(
        &analysis_report_json(
            r#"[{"name":"ok","location":{"file":"src/decompress/mod.rs","line_start":1,"line_end":2,"col_start":0,"col_end":1,"byte_start":0,"byte_end":10},"signature":{"inputs":[],"outputs":[]},"metrics":{"loc_code":1,"loc_comments":0,"loc_asm":0,"inputs":0,"outputs":0,"branches":0,"loops":0,"max_loop_nesting":0,"max_if_nesting":0,"max_combined_nesting":0,"calls_unique":0,"calls_total":0,"cyclomatic":1,"cognitive":0,"early_returns":0,"goto_count":0,"unsafe_blocks":0,"binary_operators":{"add":0,"sub":0,"mul":0,"div":0,"rem":0,"shl":0,"shr":0,"bit_and":0,"bit_or":0,"bit_xor":0,"bit_not":0,"logic_and":0,"logic_or":0,"logic_not":0}},"constants":[],"calls":[],"types_used":[]}]"#,
        ),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "expected nonnegative JSON number")]
fn analysis_report_assertion_rejects_negative_halstead_metric() {
    let function = analysis_function_json("ZSTD_decompressMultiFrame")
        .replace(r#""volume":0.0"#, r#""volume":-1.0"#);

    assert_analysis_report_contents_has_functions(
        &analysis_report_json(&format!("[{function}]")),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "duplicate JSON field functions")]
fn analysis_report_assertion_rejects_duplicate_functions_field() {
    assert_analysis_report_contents_has_functions(
        r#"{"schema_version":4,"language":"rust","source_file":"src/decompress","source_hash":"hash","functions":[{"name":"ok"}],"functions":[]}"#,
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "duplicate JSON field generated_at")]
fn analysis_report_assertion_rejects_unqueried_duplicate_field() {
    assert_analysis_report_contents_has_functions(
        r#"{"schema_version":4,"language":"rust","source_file":"src/decompress","source_hash":"hash","generated_at":1,"generated_at":2,"functions":[{"name":"ok"}]}"#,
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "missing top-level JSON field schema_version")]
fn analysis_report_assertion_requires_schema_version() {
    assert_analysis_report_contents_has_functions(
        r#"{"language":"rust","source_file":"src/decompress","source_hash":"hash","functions":[{"name":"ok"}]}"#,
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "unexpected language")]
fn analysis_report_assertion_rejects_wrong_language() {
    assert_analysis_report_contents_has_functions(
        &analysis_report_json(r#"[{"name":"ok"}]"#),
        "test analyze",
        Some("c"),
    );
}

#[test]
#[should_panic(expected = "expected JSON string for source_hash")]
fn analysis_report_assertion_rejects_non_string_source_hash() {
    assert_analysis_report_contents_has_functions(
        r#"{"schema_version":4,"language":"rust","source_file":"src/decompress","source_hash":1,"functions":[{"name":"ok"}]}"#,
        "test analyze",
        Some("rust"),
    );
}

#[test]
fn analysis_report_assertion_accepts_schema_v4_call_sites() {
    let function = analysis_function_json("ZSTD_decompressMultiFrame")
        .replace(
            r#""calls":[],"#,
            r#""calls":[{"callee":"ZSTD_decompressFrame","count":1,"span":[10,20]}],"call_sites":[{"callee":"ZSTD_decompressFrame","span":[10,20],"args":[{"kind":"param","index":0},{"kind":"const","value":{"kind":"int","value":-1,"text":"-1","span":[15,17]}},{"kind":"nestedcall","callee":"helper"},{"kind":"opaque","text":"dst + pos"}],"in_loop":true,"path_cond":{"kind":"cmp","op":"lt","left":{"kind":"param","index":1},"right":{"kind":"const","value":{"kind":"int","value":0,"text":"0","span":[1,2]}}}}],"#,
        )
        .replace(
            r#""types_used":[]}"#,
            r#""types_used":[{"text":"ZSTD_DCtx"}]}"#,
        );

    assert_analysis_report_contents_has_functions(
        &analysis_report_json(&format!("[{function}]")),
        "test analyze",
        Some("rust"),
    );
}

#[test]
fn analysis_report_assertion_accepts_whitespace_char_constant() {
    let function = analysis_function_json("ZSTD_decompressMultiFrame").replace(
        r#""constants":[],"#,
        r#""constants":[{"kind":"char","value":" ","span":[1,4]}],"#,
    );

    assert_analysis_report_contents_has_functions(
        &analysis_report_json(&format!("[{function}]")),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "constant entry 0 contained empty char value")]
fn analysis_report_assertion_rejects_empty_char_constant() {
    let function = analysis_function_json("ZSTD_decompressMultiFrame").replace(
        r#""constants":[],"#,
        r#""constants":[{"kind":"char","value":"","span":[1,3]}],"#,
    );

    assert_analysis_report_contents_has_functions(
        &analysis_report_json(&format!("[{function}]")),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "contained zero count")]
fn analysis_report_assertion_rejects_zero_call_count() {
    let function = analysis_function_json("ZSTD_decompressMultiFrame").replace(
        r#""calls":[],"#,
        r#""calls":[{"callee":"ZSTD_decompressFrame","count":0,"span":[10,20]}],"#,
    );

    assert_analysis_report_contents_has_functions(
        &analysis_report_json(&format!("[{function}]")),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "call_site entry 0 contained empty callee")]
fn analysis_report_assertion_rejects_empty_call_site_callee() {
    let function = analysis_function_json("ZSTD_decompressMultiFrame").replace(
        r#""calls":[],"#,
        r#""calls":[],"call_sites":[{"callee":"","span":[10,20],"args":[]}],"#,
    );

    assert_analysis_report_contents_has_functions(
        &analysis_report_json(&format!("[{function}]")),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "call arg entry 0 contained no kind")]
fn analysis_report_assertion_rejects_malformed_call_site_arg() {
    let function = analysis_function_json("ZSTD_decompressMultiFrame").replace(
        r#""calls":[],"#,
        r#""calls":[],"call_sites":[{"callee":"ZSTD_decompressFrame","span":[10,20],"args":[{"index":0}]}],"#,
    );

    assert_analysis_report_contents_has_functions(
        &analysis_report_json(&format!("[{function}]")),
        "test analyze",
        Some("rust"),
    );
}

#[test]
#[should_panic(expected = "constant entry 0 used unknown kind")]
fn analysis_report_assertion_rejects_unknown_constant_kind() {
    let function = analysis_function_json("ZSTD_decompressMultiFrame").replace(
        r#""constants":[],"#,
        r#""constants":[{"kind":"bytes","value":"00","span":[1,2]}],"#,
    );

    assert_analysis_report_contents_has_functions(
        &analysis_report_json(&format!("[{function}]")),
        "test analyze",
        Some("rust"),
    );
}

#[test]
fn compare_json_assertion_requires_nonempty_mapped_pair_fields() {
    let stdout = r#"[
        {
            "rust_name": "ZSTD_decompressSequencesLong_body",
            "other_name": "ZSTD_decompressSequencesLong_body",
            "total": 1.0,
            "per_metric": [["loc_code", 26.0, 113.0, 1.0]]
        }
    ]"#;

    assert_compare_json_has_mapped_pair(stdout);
}

#[test]
fn compare_top_one_assertion_requires_exactly_one_mapped_pair() {
    assert_compare_json_has_exactly_one_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":3,"per_metric":[["loc_code",1,2,3]]}]"#,
    );
}

#[test]
#[should_panic(expected = "exactly one function pair")]
fn compare_top_one_assertion_rejects_multiple_pairs() {
    assert_compare_json_has_exactly_one_mapped_pair(
        r#"[
            {"rust_name":"r0","other_name":"c0","total":3,"per_metric":[["loc_code",1,2,3]]},
            {"rust_name":"r1","other_name":"c1","total":3,"per_metric":[["loc_code",1,2,3]]}
        ]"#,
    );
}

#[test]
fn missing_structs_report_assertion_allows_extra_rust_structs() {
    assert_missing_structs_report_has_no_missing_structs(
        r#"{"missing_in_rust":[],"extra_in_rust":["RustOnlyHelper"]}"#,
    );
}

#[test]
#[should_panic(expected = "expected missing_in_rust structs to be empty")]
fn missing_structs_report_assertion_rejects_missing_structs() {
    assert_missing_structs_report_has_no_missing_structs(
        r#"{"missing_in_rust":["ZSTD_DCtx"],"extra_in_rust":[]}"#,
    );
}

#[test]
#[should_panic(expected = "expected missing_in_rust entry 0 to be non-empty")]
fn missing_structs_report_assertion_rejects_blank_missing_struct_name() {
    assert_missing_structs_report_has_no_missing_structs(
        r#"{"missing_in_rust":["   "],"extra_in_rust":[]}"#,
    );
}

#[test]
fn compare_structs_top_one_assertion_requires_exactly_one_mapped_pair() {
    assert_compare_structs_json_has_exactly_one_mapped_pair(
        r#"[{"rust_name":"ZSTD_DCtx","other_name":"ZSTD_DCtx","total":0,"per_category":[["field_count",2,2,0],["pointer",1,1,0]]}]"#,
    );
}

#[test]
#[should_panic(expected = "exactly one struct pair")]
fn compare_structs_top_one_assertion_rejects_empty_pairs() {
    assert_compare_structs_json_has_exactly_one_mapped_pair(r#"[]"#);
}

#[test]
#[should_panic(expected = "field per_category to be non-empty")]
fn compare_structs_json_assertion_rejects_empty_categories() {
    assert_compare_structs_json_has_exactly_one_mapped_pair(
        r#"[{"rust_name":"ZSTD_DCtx","other_name":"ZSTD_DCtx","total":0,"per_category":[]}]"#,
    );
}

#[test]
#[should_panic(expected = "to use a known category")]
fn compare_structs_json_assertion_rejects_unknown_category() {
    assert_compare_structs_json_has_exactly_one_mapped_pair(
        r#"[{"rust_name":"ZSTD_DCtx","other_name":"ZSTD_DCtx","total":0,"per_category":[["bytes",1,1,0]]}]"#,
    );
}

#[test]
#[should_panic(expected = "expected nonnegative JSON number")]
fn compare_structs_json_assertion_rejects_negative_total() {
    assert_compare_structs_json_has_exactly_one_mapped_pair(
        r#"[{"rust_name":"ZSTD_DCtx","other_name":"ZSTD_DCtx","total":-1,"per_category":[["field_count",1,1,0]]}]"#,
    );
}

#[test]
#[should_panic(expected = "category 1 to be unique")]
fn compare_structs_json_assertion_rejects_duplicate_category() {
    assert_compare_structs_json_has_exactly_one_mapped_pair(
        r#"[{"rust_name":"ZSTD_DCtx","other_name":"ZSTD_DCtx","total":0,"per_category":[["field_count",1,1,0],["field_count",2,2,0]]}]"#,
    );
}

#[test]
#[should_panic(expected = "compare-structs total to equal summed contributions")]
fn compare_structs_json_assertion_rejects_total_contribution_mismatch() {
    assert_compare_structs_json_has_exactly_one_mapped_pair(
        r#"[{"rust_name":"ZSTD_DCtx","other_name":"ZSTD_DCtx","total":1,"per_category":[["field_count",2,2,0],["pointer",1,1,0]]}]"#,
    );
}

#[test]
#[should_panic(expected = "expected compare result element 0 to be an object")]
fn compare_json_assertion_rejects_non_object_result() {
    assert_compare_json_has_mapped_pair(r#"[["rust_name", "other_name", "per_metric"]]"#);
}

#[test]
#[should_panic(expected = "field rust_name to be a string")]
fn compare_json_assertion_rejects_non_string_name() {
    assert_compare_json_has_mapped_pair(r#"[{"rust_name":1,"other_name":"c","per_metric":[]}]"#);
}

#[test]
#[should_panic(expected = "field per_metric to be an array")]
fn compare_json_assertion_rejects_non_array_metrics() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":1,"per_metric":{}}]"#,
    );
}

#[test]
#[should_panic(expected = "field rust_name to be non-empty")]
fn compare_json_assertion_rejects_empty_name() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"","other_name":"c","total":1,"per_metric":[["loc_code",1,2,3]]}]"#,
    );
}

#[test]
#[should_panic(expected = "field rust_name to be non-empty")]
fn compare_json_assertion_rejects_blank_name() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"   ","other_name":"c","total":1,"per_metric":[["loc_code",1,2,3]]}]"#,
    );
}

#[test]
#[should_panic(expected = "field per_metric to be non-empty")]
fn compare_json_assertion_rejects_empty_metrics() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":1,"per_metric":[]}]"#,
    );
}

#[test]
#[should_panic(expected = "metric 0 to be an array")]
fn compare_json_assertion_rejects_non_array_metric_entry() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":1,"per_metric":[{}]}]"#,
    );
}

#[test]
#[should_panic(expected = "field total to be a number")]
fn compare_json_assertion_rejects_non_number_total() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":"1","per_metric":[["loc_code",1,2,3]]}]"#,
    );
}

#[test]
#[should_panic(expected = "expected nonnegative JSON number")]
fn compare_json_assertion_rejects_negative_total() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":-1,"per_metric":[["loc_code",1,2,3]]}]"#,
    );
}

#[test]
#[should_panic(expected = "expected finite JSON number")]
fn compare_json_assertion_rejects_non_finite_total() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":1e999,"per_metric":[["loc_code",1,2,3]]}]"#,
    );
}

#[test]
#[should_panic(expected = "to have 4 fields")]
fn compare_json_assertion_rejects_short_metric_tuple() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":1,"per_metric":[["loc_code",1,2]]}]"#,
    );
}

#[test]
#[should_panic(expected = "metric 0 field 2 to be a number")]
fn compare_json_assertion_rejects_non_number_metric_value() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":1,"per_metric":[["loc_code",1,"2",3]]}]"#,
    );
}

#[test]
#[should_panic(expected = "expected nonnegative JSON number")]
fn compare_json_assertion_rejects_negative_metric_value() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":1,"per_metric":[["loc_code",1,-2,3]]}]"#,
    );
}

#[test]
#[should_panic(expected = "expected finite JSON number")]
fn compare_json_assertion_rejects_non_finite_metric_value() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":1,"per_metric":[["loc_code",1,2,1e999]]}]"#,
    );
}

#[test]
#[should_panic(expected = "metric 0 name to be a string")]
fn compare_json_assertion_rejects_non_string_metric_name() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":1,"per_metric":[[1,1,2,3]]}]"#,
    );
}

#[test]
#[should_panic(expected = "metric 0 name to be non-empty")]
fn compare_json_assertion_rejects_empty_metric_name() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":1,"per_metric":[["",1,2,3]]}]"#,
    );
}

#[test]
#[should_panic(expected = "to use a known metric")]
fn compare_json_assertion_rejects_unknown_metric_name() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":1,"per_metric":[["loc",1,2,3]]}]"#,
    );
}

#[test]
#[should_panic(expected = "metric 1 to be unique")]
fn compare_json_assertion_rejects_duplicate_metric_name() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":6,"per_metric":[["loc_code",1,2,3],["loc_code",2,3,3]]}]"#,
    );
}

#[test]
#[should_panic(expected = "compare total to equal summed contributions")]
fn compare_json_assertion_rejects_total_contribution_mismatch() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":1,"per_metric":[["loc_code",1,2,3]]}]"#,
    );
}

#[test]
#[should_panic(expected = "compare result 1 field other_name to be non-empty")]
fn compare_json_assertion_validates_later_pairs() {
    assert_compare_json_has_mapped_pair(
        r#"[
            {"rust_name":"r0","other_name":"c0","total":3,"per_metric":[["loc_code",1,2,3]]},
            {"rust_name":"r1","other_name":"","total":3,"per_metric":[["loc_code",1,2,3]]}
        ]"#,
    );
}

#[test]
#[should_panic(expected = "metric 1 field 3 to be a number")]
fn compare_json_assertion_validates_later_metrics() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":1,"per_metric":[["loc_code",1,2,3],["calls_total",1,2,"3"]]}]"#,
    );
}

#[test]
#[should_panic(expected = "duplicate JSON field rust_name")]
fn compare_json_assertion_rejects_duplicate_pair_fields() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","rust_name":"spoof","other_name":"c","total":1,"per_metric":[["loc_code",1,2,3]]}]"#,
    );
}

#[test]
#[should_panic(expected = "duplicate JSON field total")]
fn compare_json_assertion_rejects_unqueried_duplicate_pair_fields() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":1,"total":2,"per_metric":[["loc_code",1,2,3]]}]"#,
    );
}

#[test]
#[should_panic(expected = "expected ','")]
fn compare_json_assertion_rejects_invalid_tokens_inside_object() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":1,"per_metric":[["loc_code",1,2,3]] invalid}]"#,
    );
}

#[test]
#[should_panic(expected = "expected complete JSON ccc JSON object")]
fn json_array_field_contents_rejects_trailing_data() {
    assert_json_array_field_empty(r#"{"missing_in_rust": []} trailing"#, "missing_in_rust");
}

#[test]
#[should_panic(expected = "expected complete JSON ccc compare result")]
fn compare_json_assertion_rejects_trailing_data() {
    assert_compare_json_has_mapped_pair(
        r#"[{"rust_name":"r","other_name":"c","total":1,"per_metric":[]}] trailing"#,
    );
}

#[test]
#[should_panic(expected = "invalid JSON unicode surrogate pair")]
fn json_parser_rejects_unpaired_unicode_surrogate() {
    parse_json(r#"{"name":"\ud800"}"#, "test JSON object");
}

#[test]
#[ignore = "slow parity harness over translated decompression modules"]
fn ccc_decompression_audit_has_no_missing_rust_functions() {
    let ccc = comparator_bin();
    assert!(
        ccc.is_file(),
        "ccc-rs binary not found at {}",
        ccc.display()
    );

    let workdir = workdir();
    fs::create_dir_all(&workdir).expect("create comparator workdir");
    let rust_json = workdir.join("rust.json");
    let c_json = workdir.join("c.json");
    let mapping = repo_root().join("ccc_mapping.toml");
    assert!(
        mapping.is_file(),
        "ccc mapping not found at {}",
        mapping.display()
    );

    analyze(&ccc, "src/decompress", "rust", &rust_json);
    analyze(&ccc, "zstd/lib/decompress", "c", &c_json);

    let missing = run_checked(
        Command::new(&ccc)
            .current_dir(repo_root())
            .arg("missing")
            .arg(&rust_json)
            .arg(&c_json)
            .arg("--mapping")
            .arg(&mapping)
            .arg("--format")
            .arg("json"),
        "ccc missing",
    );
    let missing_stdout = String::from_utf8(missing.stdout).expect("utf8 ccc missing output");
    assert_missing_report_has_no_missing_functions(&missing_stdout);

    let compare = run_checked(
        Command::new(&ccc)
            .current_dir(repo_root())
            .arg("compare")
            .arg(&rust_json)
            .arg(&c_json)
            .arg("--mapping")
            .arg(&mapping)
            .arg("--format")
            .arg("json")
            .arg("--top")
            .arg("1"),
        "ccc compare",
    );
    let compare_stdout = String::from_utf8(compare.stdout).expect("utf8 ccc compare output");
    assert_compare_json_has_exactly_one_mapped_pair(&compare_stdout);

    let missing_structs = run_checked(
        Command::new(&ccc)
            .current_dir(repo_root())
            .arg("missing-structs")
            .arg(&rust_json)
            .arg(&c_json)
            .arg("--mapping")
            .arg(&mapping)
            .arg("--format")
            .arg("json"),
        "ccc missing-structs",
    );
    let missing_structs_stdout =
        String::from_utf8(missing_structs.stdout).expect("utf8 ccc missing-structs output");
    assert_missing_structs_report_has_no_missing_structs(&missing_structs_stdout);

    let compare_structs = run_checked(
        Command::new(&ccc)
            .current_dir(repo_root())
            .arg("compare-structs")
            .arg(&rust_json)
            .arg(&c_json)
            .arg("--mapping")
            .arg(&mapping)
            .arg("--format")
            .arg("json")
            .arg("--top")
            .arg("1"),
        "ccc compare-structs",
    );
    let compare_structs_stdout =
        String::from_utf8(compare_structs.stdout).expect("utf8 ccc compare-structs output");
    assert_compare_structs_json_has_exactly_one_mapped_pair(&compare_structs_stdout);
}

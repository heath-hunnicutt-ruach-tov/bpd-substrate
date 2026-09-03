:- protocol(citation_markdown_emitterp).
    :- public(emit_markdown_bibliography/2).
    :- public(citations_to_markdown/3).
:- end_protocol.

:- object(citation_markdown_emitter,
    implements(citation_markdown_emitterp)).

    :- uses(list, [member/2]).



:- uses(citation_extractor, [extract_citations/3, extract_citations_grouped/2]).

%% Truth Flow citation annotations — this module dogfoods its own discipline.
:- discontiguous(cites/2).
:- discontiguous(no_citation_needed/1).


%% ────────────────────────────────────────────────────────────────────────
%% emit_markdown_bibliography(+SourcePath, +OutputPath)
%%
%% Reads the source file, extracts its citations, renders them as Markdown,
%% writes to OutputPath. Creates parent directories implicitly via open/3.
%% ────────────────────────────────────────────────────────────────────────
cites(emit_markdown_bibliography/2, [
    cite(authority:internal(truth_flow_bpd_integration_design),
         section(layer_4_markdown_documentation_emission))
]).
emit_markdown_bibliography(SourcePath, OutputPath) :-
    citations_to_markdown(SourcePath, SourcePath, Markdown),
    setup_call_cleanup(
        open(OutputPath, write, Out),
        write(Out, Markdown),
        close(Out)
    ).


%% citations_to_markdown(+SourcePath, +Heading, -Markdown)
%%
%% Returns the Markdown string for a source file's citations.
%% Heading is included in the top-level # heading (typically the source path).
cites(citations_to_markdown/3, [
    cite(authority:internal(truth_flow_bpd_integration_design),
         section(layer_4_markdown_documentation_emission))
]).
citations_to_markdown(SourcePath, Heading, Markdown) :-
    extract_citations_grouped(SourcePath, Grouped),
    sort(Grouped, GroupedSorted),
    render_bibliography(Heading, GroupedSorted, Markdown).


%% render_bibliography(+Heading, +Entries, -Markdown)
no_citation_needed(render_bibliography/3).
render_bibliography(Heading, Entries, Markdown) :-
    findall(Section,
            ( member(Entry, Entries),
              render_entry(Entry, Section)
            ),
            Sections),
    atomic_list_concat(Sections, '', Body),
    format(atom(Markdown),
"# Citations for ~w

This bibliography is generated from the cites/2 annotations in the source
file by the Truth Flow citation_markdown_emitter. The annotations
themselves are the substrate-of-record; this Markdown is a projection
of that substrate into a human-readable form.

See also: docs/truth-flow-bpd-integration-design.md for the architectural
design that produces this output.

~w",
           [Heading, Body]).


%% render_entry(+PredicateCitations, -Section)
no_citation_needed(render_entry/2).
render_entry(predicate_citations(predicate(Functor/Arity), exempt), Section) :-
    format(atom(Section),
"## `~w/~w` (no citation needed)

Marked as `no_citation_needed/1` in the source. Internal helper or glue
code whose absence of citation was a deliberate decision rather than an
oversight.

",
           [Functor, Arity]).

render_entry(predicate_citations(predicate(Functor/Arity), Citations), Section) :-
    Citations \== exempt,
    findall(Line, ( member(C, Citations), render_citation(C, Line) ), Lines),
    atomic_list_concat(Lines, '', LinesText),
    format(atom(Section),
"## `~w/~w`

~w
",
           [Functor, Arity, LinesText]).


%% render_citation(+CitationExpression, -Line)
%%
%% Renders one citation expression as a single Markdown bullet line.
%% Handles both bare authorities and locator-wrapped cite/2 forms.
no_citation_needed(render_citation/2).
render_citation(cite(Authority, Locator), Line) :-
    !,
    format(atom(AuthStr), "~w", [Authority]),
    format(atom(LocStr), "~w", [Locator]),
    format(atom(Line), "- `~w` — `~w`\n", [AuthStr, LocStr]).

render_citation(Authority, Line) :-
    format(atom(AuthStr), "~w", [Authority]),
    format(atom(Line), "- `~w`\n", [AuthStr]).

:- end_object.

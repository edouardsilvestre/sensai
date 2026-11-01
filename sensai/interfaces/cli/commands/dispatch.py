from sensai.interfaces.cli.commands.answers import (
    cache_command,
    example_command,
    list_examples,
    reasoning_command,
    stats_command,
)
from sensai.interfaces.cli.commands.conversation import (
    branch_command,
    branches_command,
    context_command,
    edit_command,
    history_command,
    json_command,
    new_command,
    resume_command,
    review_command,
    sessions_command,
)
from sensai.interfaces.cli.commands.document import document_command, export_command
from sensai.interfaces.cli.commands.personas import list_personas, persona_command, profile_command
from sensai.interfaces.cli.commands.prompts import list_prompts, prompt_command
from sensai.interfaces.cli.commands.schedule import reports_command, schedule_command
from sensai.interfaces.cli.commands.search import filter_command, ingest_command, show_dropped, show_excerpts
from sensai.interfaces.cli.commands.settings import (
    approval_command,
    feature_command,
    files_command,
    guardrails_command,
    list_features,
    tools_command,
)
from sensai.interfaces.cli.commands.sharing import share_command
from sensai.interfaces.cli.output import error
from sensai.interfaces.cli.session import Session

EXIT_COMMANDS = {"/exit", "/quit"}
HELP = """\
Commands:
  /ingest PATH...         index files or folders so that answers can draw on them
  /sources                show the excerpts retrieved for the last message
  /sources dropped        show the excerpts found but dropped as not relevant enough
  /filter [FILTER|clear]  search only some documents, like after:2023-03 before:2023-07 from:dupont kind:email
  /tools                  list the tools the model can call
  /files                  show the folders the model may read and write
  /context                show the context usage and the summary of older turns
  /sessions               list the saved conversations
  /resume [ID]            continue a saved conversation (the last one without ID)
  /new                    start a new conversation, the current one stays saved
  /history                list the messages of this branch with their numbers
  /edit N [TEXT]          replace message N in a new branch; the original conversation stays in its branch
  /branches               list the branches of this conversation
  /branch N               continue branch N
  /profile                show the instructions the model gets about you
  /profile add TEXT       add an instruction to your profile
  /profile remove N       remove instruction N from your profile
  /document               show the working document the model keeps up to date
  /document undo          undo the changes made to the document by the last answer
  /document export PATH   write the document to a .md, .json, .pdf or .html file
  /export PATH            write the conversation to a .md, .json, .pdf or .html file
  /share                  list the public links
  /share conversation     publish a frozen copy of this conversation on a link anyone can open
  /share document         publish a frozen copy of the document on a link anyone can open
  /share remove N         take link N down
  /features               list the features and whether they are on
  /feature NAME on|off    turn a feature on or off for this session
  /personas               list the personas
  /persona [ID]           show the active persona, or switch to another one (the history is kept)
  /review QUESTION        the investigator answers, the fact-checker checks, the editor writes the text
  /json QUESTION          answer in JSON: the answer, its claims with their excerpts, the entities and the events
  /guardrails             list the rules checking your messages, the answers and the tool calls
  /approval [MODE]        show or change when to ask before running a tool: critical, all or off
  /reasoning [N]          list the questions reasoned about, or show the steps of question N
  /cache [clear]          show the answers kept to reuse, or remove them all
  /examples               list the answers kept as examples, * marks those shown for the last answer
  /example add|remove N   keep the last answer as an example for similar questions, or remove example N
  /stats [DAYS|clear]     show the metrics of the last DAYS days (7 by default), or clear the log
  /prompts                list the versions of the system prompt, * marks the one in use
  /prompt show|use N      read version N of the system prompt, or use it from now on (roll back)
  /prompt diff N M        show what changed between versions N and M
  /prompt compare N M     ask the test cases with versions N and M, and compare the answers, time and tokens
  /schedule               list the tasks run on their own by python -m sensai schedule
  /schedule add WHEN TASK add a task: every 6h, daily 08:00 or weekly mon 08:00, --allow TOOL for a critical tool
  /schedule remove N      remove task N, its reports stay
  /reports [N]            list the reports of the scheduled tasks, or read report N
  /exit, /quit            leave (or press Ctrl+D)
Ctrl+C during an answer stops it: what was written is kept, and your next message redirects it."""


def handle_command(session: Session, line: str) -> bool:
    name, _, argument = line.strip().partition(" ")
    argument = argument.strip()
    if name in EXIT_COMMANDS:
        return False
    if name == "/help":
        print(HELP)
    elif name == "/sources":
        if argument == "dropped":
            show_dropped(session.last_dropped)
        elif argument:
            error("Usage: /sources [dropped]")
        else:
            show_excerpts(session.last_sources)
    elif name == "/filter":
        filter_command(session, argument)
    elif name == "/tools":
        tools_command(session)
    elif name == "/files":
        files_command(session)
    elif name == "/ingest":
        ingest_command(session, argument)
    elif name == "/context":
        context_command(session)
    elif name == "/sessions":
        sessions_command(session)
    elif name == "/resume":
        resume_command(session, argument)
    elif name == "/new":
        new_command(session)
    elif name == "/history":
        history_command(session)
    elif name == "/edit":
        edit_command(session, argument)
    elif name == "/branches":
        branches_command(session)
    elif name == "/branch":
        branch_command(session, argument)
    elif name == "/profile":
        profile_command(session, argument)
    elif name == "/document":
        document_command(session, argument)
    elif name == "/export":
        export_command(session, argument)
    elif name == "/share":
        share_command(session, argument)
    elif name == "/guardrails":
        guardrails_command(session)
    elif name == "/approval":
        approval_command(session, argument.lower())
    elif name == "/reasoning":
        reasoning_command(session, argument)
    elif name == "/cache":
        cache_command(session, argument.lower())
    elif name == "/examples":
        list_examples(session)
    elif name == "/example":
        example_command(session, argument.lower())
    elif name == "/stats":
        stats_command(session, argument.lower())
    elif name == "/prompts":
        list_prompts(session)
    elif name == "/prompt":
        prompt_command(session, argument)
    elif name == "/schedule":
        schedule_command(session, argument)
    elif name == "/reports":
        reports_command(session, argument)
    elif name == "/features":
        list_features(session)
    elif name == "/feature":
        feature_command(session, argument)
    elif name == "/personas":
        list_personas(session)
    elif name == "/persona":
        persona_command(session, argument)
    elif name == "/review":
        review_command(session, argument)
    elif name == "/json":
        json_command(session, argument)
    else:
        error(f"Unknown command {name}. Type /help for the list.")
    return True

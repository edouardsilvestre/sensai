from sensai.core.export.report import YOU, Publication, prepare_publication, to_html
from sensai.core.export.shares import Share, ShareError, public_link
from sensai.core.guardrails.rules import Hit
from sensai.interfaces.cli.commands.document import report_of
from sensai.interfaces.cli.output import error, plural, shorten
from sensai.interfaces.cli.session import Session, confirm

SHARE_KINDS = ("conversation", "document")
SHARE_USAGE = "Usage: /share [conversation | document | remove N]"
SERVE_HINT = "The links answer while python -m sensai publish runs."


def share_refusal(session: Session, kind: str) -> str | None:
    memory = session.agent.memory
    if not session.config.features["sharing"]:
        return "The sharing feature is off: /feature sharing on to publish on a public link."
    if kind == "document":
        return "The document is empty: nothing to publish." if memory.document.is_empty else None
    return None if memory.turns or memory.summary else "No message to publish yet."


def publication_of(session: Session, kind: str) -> Publication:
    return prepare_publication(report_of(session, kind), session.agent.active_guardrails)


def blocked_message(hit: Hit) -> str:
    return " ".join(filter(None, [f"Not published: the rule '{hit.rule}' found {hit.found}.", hit.message]))


def publication_lines(session: Session, publication: Publication) -> list[str]:
    """What the user reads before publishing: what the page holds and what the rules changed in it."""
    report = publication.report
    messages = sum(part.author == YOU for part in report.parts)
    lines = [f"Page: {report.title}" + (f" ({plural(messages, 'message')})" if messages else "")]
    if not session.config.features["guardrails"]:
        lines.append("The guardrails are off: nothing was checked or redacted.")
    for hit in publication.hits:
        done = "redacted" if hit.action == "redact" else "found, and left in the page"
        lines.append(" ".join(filter(None, [f"Rule '{hit.rule}': {hit.count} {done} ({hit.found}).", hit.message])))
    if publication.sources:
        lines.append(f"Sources named: {'; '.join(publication.sources)}")
    return lines


def publish(session: Session, kind: str, publication: Publication) -> Share:
    return session.shares.add(kind, publication.report.title, to_html(publication.report))


def list_shares(session: Session) -> None:
    shares = session.shares.list()
    if not shares:
        print("No public link yet. /share conversation or /share document to publish one.")
        return
    for number, share in enumerate(shares, start=1):
        print(f"{number}. {share.created:%Y-%m-%d %H:%M}  {share.kind}  {shorten(share.title)}")
        print(f"   {public_link(session.config, share)}")
    print(f"/share remove N to take a link down. {SERVE_HINT}")


def remove_share(session: Session, value: str) -> None:
    shares = session.shares.list()
    if not value.isdecimal():
        error(SHARE_USAGE)
    elif not 1 <= int(value) <= len(shares):
        error(f"No link {value}: /share lists them.")
    else:
        share = shares[int(value) - 1]
        try:
            session.shares.remove(share.token)
        except ShareError as e:
            error(str(e))
            return
        print(f"Link {value} is down: {shorten(share.title)}")


def publish_command(session: Session, kind: str) -> None:
    refusal = share_refusal(session, kind)
    if refusal:
        print(refusal)
        return
    publication = publication_of(session, kind)
    if publication.blocked:
        error(blocked_message(publication.blocked))
        return
    for line in publication_lines(session, publication):
        print(line)
    print("Anyone who gets the link can read this page, without logging in.")
    if not confirm("Publish? [y]es / [n]o: "):
        print("Nothing was published.")
        return
    try:
        share = publish(session, kind, publication)
    except ShareError as e:
        error(str(e))
        return
    print(f"Published: {public_link(session.config, share)}")
    print(f"The page stays as it is now: the next messages are not added. {SERVE_HINT}")


def share_command(session: Session, argument: str) -> None:
    action, _, value = argument.partition(" ")
    if not action:
        list_shares(session)
    elif action == "remove":
        remove_share(session, value.strip())
    elif action in SHARE_KINDS and not value:
        publish_command(session, action)
    else:
        error(SHARE_USAGE)

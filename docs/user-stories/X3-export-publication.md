# X3 Export & Publication: user stories

## Keep a copy of the work

As an investigative journalist closing a stage of a case, I want to export the conversation to a file, so that it is archived with the case and can be read without Sensai.

Acceptance criteria:
- `/export PATH` writes the conversation; the end of the name says the format: `.md`, `.json`, `.pdf` or `.html`.
- Any other ending, or a folder that cannot be written, gives an error and writes nothing; a conversation without any message is not exported.
- The Markdown, the PDF and the HTML hold the active branch, with the summary of the older messages when some were condensed, and each answer with the sources it cites. When there are other branches, the CLI says so.
- The JSON is the saved conversation with all its branches and its tool calls: copied into `SENSAI_SESSIONS_DIR`, it can be resumed.
- A local export is a faithful copy: the guardrails do not change it.

## Send the working document to the editor

As an investigative journalist, I want to export the working document as a PDF, so that my editor can read the state of the case without any tool.

Acceptance criteria:
- `/document export PATH` takes the same endings; the Markdown stays exactly the document.
- The PDF has the title, the sections, the lists, the bold and the code, and its pages are numbered.
- Accents, the euro sign, quotes and dashes are kept. A character the font does not have, like Cyrillic, is shown as `?`, and the last line of the PDF says how many were replaced and that the Markdown and the JSON keep them.
- The page has a button for each format, for the conversation and for the document.

## Share a page with someone who has no Sensai

As an investigative journalist, I want to publish a conversation or the document on a link, so that a colleague can read it in a browser without an account or an installation.

Acceptance criteria:
- With the `sharing` feature on (off by default), `/share conversation` or `/share document` shows what the page holds, asks for a confirmation, and gives a link; on the web page, `Create a link` shows the same, with `Publish`, `See the page` and `Cancel`.
- `python -m sensai publish` serves the pages; anyone with a link can read its page, on a phone too, without logging in.
- Refusing, or the end of the input, publishes nothing.
- The page is a frozen copy: the messages that come after are not added to it.
- A page created while the server is running is served without restarting it.

## Publish without giving my source away

As an investigative journalist protecting a source, I want what is published to go through my rules, so that a public page never names what a chat would not.

Acceptance criteria:
- The page is checked by every rule on `input`, `output` and the new `publish` stage: a `redact` rule redacts it, a `warn` rule is reported, and a `block` rule stops the publication with the name of the rule.
- By default, personal data is redacted from a public page, like it is blocked from a web search.
- Before confirming, I see how many matches each rule found, and the sources the page names.
- The folders of the sources stay on this machine: the page only names the file. A file named after a protected name, like `claire_benoit.eml`, is replaced by `[source hidden by a rule]`.
- The tool calls, the profile and the system prompt are never published.
- With the guardrails off, the preview says that nothing was checked.

## Keep the chat private

As an investigative journalist, I want the public pages to be served apart from the chat, so that publishing one page opens nothing else.

Acceptance criteria:
- The chat keeps listening on `127.0.0.1` only. `python -m sensai publish` is another process, that answers `/shared/<link>` and nothing else: no conversation, no document, no API.
- It starts without the model, and it does not load the chat.
- A link is random and cannot be guessed; any other address gets a 404.
- A page has no script and loads nothing, and its headers keep it out of search engines and caches, and keep the link from reaching the next site.
- The links are not written to any log.

## Take a page down

As an investigative journalist who published too much, I want to take a link down at once, so that nobody can read the page any more.

Acceptance criteria:
- `/share` lists the links, and `/share remove N` takes one down; on the web page, `Take down` does the same.
- The link stops answering at once, without restarting the server, and says that the page is no longer shared.
- Taking a link down works with the feature off, and on the web page during an answer.

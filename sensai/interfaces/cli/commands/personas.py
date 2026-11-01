from sensai.core.memory.profile import ProfileError
from sensai.interfaces.cli.output import error
from sensai.interfaces.cli.session import Session
from sensai.interfaces.cli.startup import unknown_persona


def profile_command(session: Session, argument: str) -> None:
    profile = session.profile
    if profile is None:
        error("The profile could not be read at startup: fix the file and restart Sensai.")
        return
    action, _, value = argument.partition(" ")
    try:
        if action == "add":
            profile.add(value)
            print(f"Added as instruction {len(profile.entries)}.")
        elif action == "remove":
            if not value.strip().isdecimal():
                error("Usage: /profile remove N")
                return
            print(f"Removed: {profile.remove(int(value))}")
        elif action:
            error("Usage: /profile [add TEXT | remove N]")
            return
    except ProfileError as e:
        error(str(e))
        return
    session.agent.memory.profile = profile.text()
    if not action:
        show_profile(session)
    elif not session.config.features["profile"]:
        print("The profile feature is off: /feature profile on to send it to the model.")


def show_profile(session: Session) -> None:
    profile = session.profile
    if not profile.entries:
        print(f"Your profile is empty ({profile.path}). Add what the model should always know, for example:")
        print("  /profile add I work on the Alvor Consulting case for Le Monde; answer in French.")
        return
    state = "" if session.config.features["profile"] else " (not sent: the profile feature is off)"
    print(f"Profile ({profile.path}){state}:")
    for number, entry in enumerate(profile.entries, start=1):
        print(f"  {number}. {entry}")


def list_personas(session: Session) -> None:
    if not session.personas:
        print(f"No persona in {session.config.personas_dir}.")
        return
    active = session.agent.persona
    width = max(len(persona_id) for persona_id in session.personas)
    for persona in session.personas.values():
        marker = "*" if active and active.id == persona.id else " "
        print(f" {marker} {persona.id:<{width}}  {persona.name}: {persona.role}")


def persona_command(session: Session, argument: str) -> None:
    agent = session.agent
    if not argument:
        if agent.persona:
            print(f"Active persona: {agent.persona.id} ({agent.persona.name}). /personas to list them.")
        else:
            print("No active persona. /personas to list them.")
        return
    persona = session.personas.get(argument)
    if persona is None:
        error(unknown_persona(session.personas, argument))
        return
    agent.set_persona(persona)
    print(f"Switched to {persona.name}. The conversation history is kept.")

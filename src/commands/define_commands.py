import argparse
import pkgutil
import importlib
from .base_command import BaseCommand  # Make sure BaseCommand is accessible

def define_commands():
    parser = argparse.ArgumentParser(description="CLI tool with automatic command discovery.")
    subparsers = parser.add_subparsers(dest='command', help='Subcommand to run')

    # Discover and load all command modules in the 'commands' package
    command_classes = []
    for finder, name, ispkg in pkgutil.iter_modules(['commands']):
        if not ispkg:
            # Import the module
            module = importlib.import_module(f'commands.{name}')
            # Scan for any class that is a subclass of BaseCommand
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, BaseCommand) and attr is not BaseCommand:
                    command_classes.append(attr())

    # Set up the parser for each command class found
    for command in command_classes:
        cmd_name = command.__class__.__name__.replace('Command', '').lower()
        cmd_help = command.__doc__ or 'No description available'
        command.create_parser(subparsers, cmd_name, cmd_help)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(**vars(args))
    else:
        # parser.print_help()
        pass

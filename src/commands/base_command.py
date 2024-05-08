
class BaseCommand:
    def create_parser(self, subparsers, command_name, help_text):
        parser = subparsers.add_parser(command_name, help=help_text)
        self.add_arguments(parser)
        parser.set_defaults(func=self.handle)
        return parser

    def add_arguments(self, parser):
        pass

    def handle(self, *args, **options):
        raise NotImplementedError("Subclasses must implement handle() method.")

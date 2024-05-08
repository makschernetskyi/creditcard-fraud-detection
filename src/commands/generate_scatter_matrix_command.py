from .base_command import BaseCommand
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import services



class GenerateScatterMatrixCommand(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument('--path', type=str, default='./plots/', help='path to save the matrix')

    def handle(self, **options):
        path = options['path']
        services.generate_scatter_matrix(path)






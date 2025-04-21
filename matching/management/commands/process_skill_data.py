import os
import pickle
from django.core.management.base import BaseCommand
from django.conf import settings

class Command(BaseCommand):
    help = 'Loads and validates skill label encoders.'

    def load_encoder_safe(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.stdout.write(f"✅ Loaded encoder from {file_path}")
                return pickle.load(f)
        else:
            self.stdout.write(self.style.ERROR(f"❌ Encoder file not found at {file_path}"))
            return None

    def handle(self, *args, **kwargs):
        # Get the absolute paths for the encoders
        encoder_skill_name_path = os.path.join(settings.BASE_DIR, 'skill_assessment', 'encoders', 'label_encoder_skill_name.pkl')
        encoder_skill_type_path = os.path.join(settings.BASE_DIR, 'skill_assessment', 'encoders', 'label_encoder_skill_type.pkl')

        # Print paths for debugging
        self.stdout.write(f"Checking encoder paths:")
        self.stdout.write(f"Skill name encoder path: {encoder_skill_name_path}")
        self.stdout.write(f"Skill type encoder path: {encoder_skill_type_path}")

        # Load encoders
        label_encoder_skill_name = self.load_encoder_safe(encoder_skill_name_path)
        label_encoder_skill_type = self.load_encoder_safe(encoder_skill_type_path)

        # Validate
        if label_encoder_skill_name is None or label_encoder_skill_type is None:
            self.stdout.write(self.style.ERROR("❌ Required encoder files are missing."))
        else:
            self.stdout.write(self.style.SUCCESS("✅ Both encoders loaded successfully!"))

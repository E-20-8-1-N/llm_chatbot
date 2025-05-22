from rest_framework import serializers

class ChatInputSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=1000)
    file_path = serializers.CharField(max_length=1000, help_text="Relative path to the file from the user data root (e.g., '/mnt/photos/my_image.jpg')")

class ChatResponseSerializer(serializers.Serializer):
    answer = serializers.CharField()
    file_queried = serializers.CharField()
    question_asked = serializers.CharField()
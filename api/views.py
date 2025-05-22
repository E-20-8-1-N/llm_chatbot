from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ChatInputSerializer, ChatResponseSerializer
from .llm_handler import process_query_with_llm, llm # Import llm to check status
import logging

logger = logging.getLogger(__name__)

class ChatbotView(APIView):
    def post(self, request, *args, **kwargs):
        if llm is None: # Check if LLM initialized correctly
            logger.error("ChatbotView: LLM is not available.")
            return Response(
                {"error": "The Language Model is not available. Please check server configuration and logs."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        serializer = ChatInputSerializer(data=request.data)
        if serializer.is_valid():
            question = serializer.validated_data['question']
            relative_file_path = serializer.validated_data['file_path']

            logger.info(f"Received chat request: Question='{question}', File='{relative_file_path}'")
            answer = process_query_with_llm(question, relative_file_path)

            response_data = {
                "answer": answer,
                "file_queried": relative_file_path,
                "question_asked": question
            }
            return Response(ChatResponseSerializer(response_data).data, status=status.HTTP_200_OK)
        logger.warning(f"Invalid chat request data: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request, *args, **kwargs):
        # Simple health check or status for GET requests
        if llm is None:
            status_message = "LLM is not initialized or failed to load. Check server logs."
            llm_status = "Unavailable"
        else:
            status_message = "Chatbot API is running. LLM appears to be loaded."
            llm_status = "Available"
        
        return Response({
            "message": status_message,
            "llm_status": llm_status,
            "usage": "Send a POST request to this endpoint with 'question' and 'file_path'."
        }, status=status.HTTP_200_OK)
from rest_framework.authentication import TokenAuthentication
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.authtoken.models import Token

class CookieTokenAuthentication(TokenAuthentication):
    def authenticate(self, request):
        # Try cookie first
        token = request.COOKIES.get('auth_token')

        # Try Authorization header next (e.g., "Bearer <token>")
        if not token:
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split("Bearer ")[1]

        # Try request body last
        if not token and hasattr(request, 'data'):
            token = request.data.get('token')

        print("Received auth_token:", token)

        if not token:
            return None

        try:
            token_obj = Token.objects.get(key=token)
        except Token.DoesNotExist:
            raise AuthenticationFailed('Invalid token')

        return (token_obj.user, token_obj)

#!/usr/bin/env python3
"""Script to update Postman collection with authentication token for all GET requests."""

import json
import re

def update_url_with_token(url_raw):
    """Update URL to include token parameter if it's a GET request."""
    # Check if token already exists in the URL
    if 'token=' in url_raw:
        return url_raw
    
    if '?' in url_raw:
        # URL already has query parameters, add token
        return f"{url_raw}&token={{auth_token}}"
    else:
        # URL has no query parameters, add token
        return f"{url_raw}?token={{auth_token}}"

def add_query_param_to_url(url_obj):
    """Add token query parameter to URL object."""
    if 'query' not in url_obj:
        url_obj['query'] = []
    
    # Check if token already exists
    token_exists = any(param.get('key') == 'token' for param in url_obj.get('query', []))
    
    if not token_exists:
        url_obj['query'].append({
            "key": "token",
            "value": "{{auth_token}}",
            "description": "Authentication token"
        })
    
    return url_obj

def update_request_with_auth(request):
    """Update a single request to include authentication."""
    # Update both GET and POST requests
    if request.get('method') in ['GET', 'POST', 'DELETE', 'PUT', 'PATCH']:
        # Update raw URL
        if 'url' in request and 'raw' in request['url']:
            request['url']['raw'] = update_url_with_token(request['url']['raw'])
        
        # Update URL object with query parameters
        if 'url' in request:
            request['url'] = add_query_param_to_url(request['url'])
    
    return request

def update_item_recursively(item):
    """Recursively update all items in the collection."""
    if 'request' in item:
        # This is a request item
        item['request'] = update_request_with_auth(item['request'])
    elif 'item' in item:
        # This is a folder item, process its children
        for sub_item in item['item']:
            update_item_recursively(sub_item)
    
    return item

def main():
    """Main function to update the Postman collection."""
    # Read the current collection
    with open('postman_collection.json', 'r') as f:
        collection = json.load(f)
    
    # Update all items recursively
    if 'item' in collection:
        for item in collection['item']:
            update_item_recursively(item)
    
    # Write the updated collection back
    with open('postman_collection.json', 'w') as f:
        json.dump(collection, f, indent=2)
    
    print("✅ Postman collection updated with authentication token for all requests")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import base64
import json
import logging
import os
import sys
import time

from googleapiclient import discovery
from googleapiclient import errors
from oauth2client.client import GoogleCredentials

DISCOVERY_URL = 'https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'
BATCH_SIZE = 10


class VisionApi:
    """Construct and use the Google Vision API service."""

    def __init__(self, api_discovery_file='vision_api.json'):
        self.logger = logging.getLogger()
        self.credentials = GoogleCredentials.get_application_default()
        self.service = discovery.build(
            'vision', 'v1', credentials=self.credentials,
            discoveryServiceUrl=DISCOVERY_URL)

    def detect_text(self, input_img, num_retries=1, max_results=6):
        """Uses the Vision API to detect text in the given file.
        """
        start = time.time()
        with open(input_img, 'rb') as image_file:
            input_img_cont = image_file.read()

        batch_request = []
        batch_request.append({
            'image': {
                'content': base64.b64encode(input_img_cont).decode('UTF-8')
            },
            'features': [{
                'type': 'TEXT_DETECTION',
                'maxResults': max_results,
            }]
        })
        request = self.service.images().annotate(
            body={'requests': batch_request})
        end = time.time()
        self.logger.info(
            'Prepare and read content locally: {}'.format(end - start))

        try:
            start = time.time()
            responses = request.execute(num_retries=num_retries)
            end = time.time()
            self.logger.info(
                'Time upload file and get response'.format(end - start))
            if 'responses' not in responses:
                return {}
            text_response = {}
            response = responses['responses']
            if 'error' in response:
                print("API Error for %s: %s" % (
                        input_img,
                        response['error']['message']
                        if 'message' in response['error']
                        else ''))
                return
            if 'textAnnotations' in response[0]:
                text_response[input_img] = response[0]['textAnnotations']
            else:
                text_response[input_img] = []

            response = [elem for elem in text_response[input_img] if \
                elem['description'].startswith('http')]
            return json.dumps(response)
        except errors.HttpError as e:
            print("Http Error for %s: %s" % (input_img, e))
        except KeyError as e2:
            print("Key error: %s" % e2)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Error: please provide an image')
        sys.exit(1)
    input_img = sys.argv[1]

    vision = VisionApi()
    response = vision.detect_text(input_img)

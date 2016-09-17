#!/usr/bin/env python

import base64
import json
import logging
import os
import re
import sys
import time

import bleach
from googleapiclient import discovery, errors
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

    def detect_text(
            self,
            input_img=None,
            base64_str=None,
            num_retries=1,
            max_results=6):
        """Uses the Vision API to detect text in the given file.
        """
        if input_img is None and base64_str is None:
            self.logger.error('One of input_img or base64_str needs to be '
                              'specified')
            return {}
        start = time.time()
        batch_request = []
        if base64_str is None:
            with open(input_img, 'rb') as image_file:
                input_img_cont = image_file.read()
            content = base64.b64encode(input_img_cont).decode('UTF-8')
        else:
            content = re.sub('^data:image/.+;base64,', '', base64_str)

        batch_request.append({
            'image': {
                'content': content,
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
                'Time upload file and get response {}'.format(end - start))
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
            if len(response) == 0:
                return {}
            if 'textAnnotations' in response[0]:
                text_response[input_img] = response[0]['textAnnotations']
            else:
                text_response[input_img] = []

            hrefs = []

            def get_href(attrs, new=False):
                hrefs.append(attrs['href'])
                return bleach.callbacks.nofollow(attrs, new)

            # contains all concatenated text
            text_response = text_response.values()[0][1:]
            response = []
            for resp in text_response:
                desc = resp['description']
                link_desc = bleach.linkify(desc, [get_href])
                if link_desc != desc:
                    href = hrefs[-1]
                    print href
                    resp['href'] = href
                    response.append(resp)

            return json.dumps(response)
        except errors.HttpError as e:
            print("Http Error for %s: %s" % (input_img, e))
        except KeyError as e2:
            print("Key error: %s" % e2)


def text_filter(data):
    text = data['description']
    match = re.search('(https?://)|(www\.)|(^@[^\.]+$)|(^.+@.+\..+)', text,
                      re.IGNORECASE)
    return bool(match)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Error: please provide an image')
        sys.exit(1)
    input_img = sys.argv[1]

    vision = VisionApi()
    response = vision.detect_text(input_img)

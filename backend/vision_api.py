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
            telephone = ""
            start_telephone_box = None
            end_telephone_box = None
            for resp in text_response:
                desc = resp['description']
                if telephone:
                    match = filter_telephone_cont(desc)
                    if match:
                        end_telephone_box = resp['boundingPoly']
                        telephone += match.group(0)
                        continue
                    else:
                        phone = process_telephone(telephone, start_telephone_box, end_telephone_box)
                        if phone:
                            response.append(phone)
                        telephone = ''
                        start_telephone_box = None
                        end_telephone_box = None

                link_desc = bleach.linkify(desc, [get_href])
                if link_desc != desc:
                    href = hrefs[-1]
                    print href
                    resp['href'] = href
                    response.append(resp)
                else:
                    match = filter_email(desc)
                    if match:
                        resp['href'] = 'mailto:' + match.group(0)
                        print(resp['href'])
                        response.append(resp)
                    else:
                        match = filter_telephone_start(desc)
                        if (match):
                            telephone += match.group(0)
                            start_telephone_box = resp['boundingPoly']

            return json.dumps(response)
        except errors.HttpError as e:
            print("Http Error for %s: %s" % (input_img, e))
        except KeyError as e2:
            print("Key error: %s" % e2)


def filter_email(text):
    match = re.search('(\S+@\S+\.\S+)', text, re.IGNORECASE)
    return match

def filter_telephone_start(text):
    match = re.match('\+[0-9]*', text)
    return match

def filter_telephone_cont(text):
    match = re.match('[0-9]+', text)
    return match

def process_telephone(telephone, start_box, end_box):
    result = {}
    if len(telephone) > 4:
        result['href'] = "intent:" + telephone + "#Intent;scheme=tel;end"
        minX = float("inf")
        maxX = float("-inf")
        minY = float("inf")
        maxY = float("-inf")
        all_vertices = start_box["vertices"] + end_box["vertices"]
        for vertex in all_vertices:
            minX = min(minX, vertex["x"])
            maxX = max(maxX, vertex["x"])
            minY = min(minY, vertex["y"])
            maxY = max(maxY, vertex["y"])

        result["boundingPoly"] = {"vertices": [
            {"x": minX, "y": minY},
            {"x": maxX, "y": minY},
            {"x": minX, "y": maxY},
            {"x": maxX, "y": maxY},
        ]}

    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Error: please provide an image')
        sys.exit(1)
    input_img = sys.argv[1]

    vision = VisionApi()
    response = vision.detect_text(input_img)

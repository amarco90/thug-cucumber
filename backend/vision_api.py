#!/usr/bin/env python

import base64
import json
import logging
import os
import re
import sys
import time

import httplib2
from collections import namedtuple

import bleach
from googleapiclient import discovery, errors
from oauth2client.client import GoogleCredentials

DISCOVERY_URL = 'https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'
BATCH_SIZE = 10


class VisionApi:
    """Construct and use the Google Vision API service."""

    def __init__(self, api_discovery_file='vision_api.json'):
        self.text_analyzer = TextAnalyzer()
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
            full_text = text_response.values()[0][0]['description']

            Entity = namedtuple('Entity', ['salience', 'name', 'wikipedia_url'])

            entities = self.text_analyzer.nl_detect(full_text)
            entity_tuples = []
            for entity in entities:
                salience = entity['salience']
                name = entity['name'].lower()
                wikipedia_url = entity['metadata'].get('wikipedia_url')

                if wikipedia_url:
                    entity_tuples.append(Entity(salience, name, wikipedia_url))

            entity_tuples.sort(reverse=True)
            print entity_tuples
            # print entities
            text_response = text_response.values()[0][1:]
            response = []
            for resp in text_response:
                desc = resp['description']
                for ent in entity_tuples:
                    if desc.lower() in ent.name:
                        resp['href'] = ent.wikipedia_url
                        response.append(resp)
                        continue
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

            return json.dumps(response)
        except errors.HttpError as e:
            print("Http Error for %s: %s" % (input_img, e))
        except KeyError as e2:
            print("Key error: %s" % e2)


class TextAnalyzer(object):
    """Construct and use the Google Natural Language API service."""

    def __init__(self, db_filename=None):
        credentials = GoogleCredentials.get_application_default()
        scoped_credentials = credentials.create_scoped(
            ['https://www.googleapis.com/auth/cloud-platform'])
        http = httplib2.Http()
        scoped_credentials.authorize(http)
        self.service = discovery.build('language', 'v1beta1', http=http)

        # This list will store the entity information gleaned from the
        # image files.
        self.entity_info = []

    def _get_native_encoding_type(self):
        """Returns the encoding type that matches Python's native strings."""
        if sys.maxunicode == 65535:
            return 'UTF16'
        else:
            return 'UTF32'

    def nl_detect(self, text):
        """Use the Natural Language API to analyze the given text string."""
        # We're only requesting 'entity' information from the Natural Language
        # API at this time.
        body = {
            'document': {
                'type': 'PLAIN_TEXT',
                'content': text,
            },
            'encodingType': self._get_native_encoding_type(),
        }
        entities = []
        try:
            request = self.service.documents().analyzeEntities(body=body)
            response = request.execute()
            entities = response['entities']
        except errors.HttpError as e:
            logging.error('Http Error: %s' % e)
        except KeyError as e2:
            logging.error('Key error: %s' % e2)
        return entities

    def extract_entity_info(self, entity):
        """Extract information about an entity."""
        type = entity['type']
        name = entity['name'].lower()
        metadata = entity['metadata']
        salience = entity['salience']
        wiki_url = metadata.get('wikipedia_url', None)
        return (type, name, salience, wiki_url)


def filter_email(text):
    match = re.search('(\S+@\S+\.\S+)', text, re.IGNORECASE)
    return match

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Error: please provide an image')
        sys.exit(1)
    input_img = sys.argv[1]

    vision = VisionApi()
    response = vision.detect_text(input_img)

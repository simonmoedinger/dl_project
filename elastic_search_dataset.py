from elasticsearch import Elasticsearch
#from datasets import 
import socket
import requests
import numpy as np
import datetime

class ElasticSearchDataset:
    def __init__(self, scroll_size=100, scroll_time='30d', doc_min_words=1100):
        address = self._get_address()
        self.es_client = Elasticsearch([address])
        cluster_settings = {
            "persistent": {
                "search.max_keep_alive": "30d"
            }
        }

        # Update cluster settings
        response = self.es_client.cluster.put_settings(body=cluster_settings)

        # Check the response status
        if 'acknowledged' in response and response['acknowledged']:
            print('Cluster settings updated successfully.')
        else:
            print(f'Error updating cluster settings: {response}')
        self.es_client.max_keep_alive = "30d"
        self.index = "content"
        
        self.mapping_function = self.identity
        self.scroll_size = scroll_size
        self.scroll_time = scroll_time
        self.scroll_id = None
        self.hits = []
        self.excluded_documents=0
        self.true_idx = -1
        
        self.doc_min_words = doc_min_words # ca 10kb text
        #medical_terms = open("/home/smoeding2/data/medizinische_begriffe_wiki_removed_frequent.txt","r")
        medical_terms = open("/home/smoeding2/data/common_medical_terms.txt","r")
        self.medical_term_set = set()

        for term in medical_terms.read().replace("\n","").split(","):
            if term.count(" ") < 3 and len(term) < 20:
                self.medical_term_set.add(term)
        
        # Initial search request
        self.total_documents = self._get_total_documents()
        #self.query = {
        #    "query": {
        #        "match_all": {}
        #    },
        #    "stored_fields": ["content", "title"]  # Specify fields to retrieve
        #}
        self.query = {
            "query": {
                "match_all": {}
            },
            "stored_fields": ["content", "title"]  # Specify fields to retrieve
        }
        self._initialize()
    def identity(self, x):
        return x
    def _get_total_documents(self):
        count_query = {
            "query": {
                "match_all": {}
            }
        }
        return self.es_client.count(index=self.index, body=count_query)['count']

    def _initialize(self):
        response = self.es_client.search(
            index=self.index,
            body=self.query,
            scroll=self.scroll_time,
            size=self.scroll_size,
        )
        self.scroll_id = response['_scroll_id']
        self.hits = response['hits']['hits']

    def __len__(self):
        return self.total_documents
    
    def map(self, function):
        self.mapping_function = function
    def __getitem__(self, idx):
        self.true_idx += 1
        if self.true_idx >= self.__len__():
                print("Epoch has ended")
        if self.true_idx%self.scroll_size == 0 and self.true_idx != 0:
            #print("Next scroll")
            response = self.es_client.scroll(scroll_id=self.scroll_id, scroll=self.scroll_time)
            self.hits = response['hits']['hits']
            time = datetime.datetime.now().strftime("%I:%M%p %B %d, %Y")
            print(f"{time} Loaded document {self.true_idx} of {self.__len__()} ({round(self.true_idx / self.__len__(),4)*100}%)")
        #print(idx%self.scroll_size)
        hit = self.hits[self.true_idx%self.scroll_size]
        content = hit['fields']['content'][0]#.replace("\n","")
        
        split = content.split()
        if len(split) > self.doc_min_words and self.medical_term_set.intersection(split):
            sentences = np.array(content.split("."))
            sentence_lengths = np.array(list(map(len,map(str.split, sentences))))
            sentences = sentences[(sentence_lengths > 4) & (sentence_lengths < 30)]
            return {"text": self.mapping_function(".".join(sentences))}
        else:
            self.excluded_documents += 1
            return self.__getitem__("Does not matter")
    
    def release_scroll(self):
        self.es_client.clear_scroll(scroll_id=self.scroll_id)
    def _get_address(self):
        def get_current_ip():
            # Get the hostname
            hostname = socket.gethostname()

            # Get the IP address associated with the hostname
            ip_address = socket.gethostbyname(hostname)

            return ip_address

        def replace_last_digit_with_one(ip_address):
            # Split the IP address into its components
            parts = ip_address.split('.')

            # Replace the last digit with '1'
            parts[-1] = '1'

            # Join the parts back to form the modified IP address
            modified_ip = '.'.join(parts)

            return modified_ip
        # Get and print the current IP address
        current_ip = get_current_ip()
        print(f"Current Container Address: {current_ip}")

        # Replace the last digit with '1' and print the modified IP address
        modified_ip = replace_last_digit_with_one(current_ip)
        print(f"Host/Bridge Address: {modified_ip}")


        def send_get_request(ip_address, port):
            url = f"http://{ip_address}:{port}"

            try:
                response = requests.get(url)
                # Check if the request was successful (status code 200)
                if response.status_code == 200:
                    print(f"GET request to {url} was successful.")
                    print("Response content:")
                    print(response.text)  # Print the content of the response
                else:
                    print(f"GET request to {url} failed with status code {response.status_code}.")
            except requests.ConnectionError:
                print(f"Failed to connect to {url}. Check if the server is running.")

        # Assuming port 9200 for the example
        port = 9200
        # Replace the last digit of the current IP with '1'
        modified_ip = replace_last_digit_with_one(current_ip)
        
        return f"http://{modified_ip}:{port}"

# Example usage:
# es = Elasticsearch(['http://your-elasticsearch-server:9200'])
# index = 'your_index'
# query = {"query": {"match_all": {}}}
# elasticsearch_dataset = ElasticsearchDataset(es, index, query)
# print(len(elasticsearch_dataset))  # Prints the total number of documents in the index
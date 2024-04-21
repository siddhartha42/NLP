from util import *
from typing import List

class Evaluation():

    def queryPrecision(self, query_doc_IDs_ordered: List[int], query_id: int, true_doc_IDs: List[int], k: int) -> float:
        """
        Computation of precision of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        query_doc_IDs_ordered : List[int]
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        query_id : int
            The ID of the query in question
        true_doc_IDs : List[int]
            The list of IDs of documents relevant to the query (ground truth)
        k : int
            The k value

        Returns
        -------
        float
            The precision value as a number between 0 and 1
        """

        relevant_docs = true_doc_IDs
        retrieved_docs = query_doc_IDs_ordered[:k]

        relevant_retrieved_docs = intersection(relevant_docs, retrieved_docs)
       
        precision = float( len(relevant_retrieved_docs) / k )
        
        return precision

    def meanPrecision(self, doc_IDs_ordered: List[List[int]], query_ids: List[int], qrels: List[dict], k: int) -> float:
        """
        Computation of precision of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        doc_IDs_ordered : List[List[int]]
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        query_ids : List[int]
            A list of IDs of the queries for which the documents are ordered
        qrels : List[dict]
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        k : int
            The k value

        Returns
        -------
        float
            The mean precision value as a number between 0 and 1
        """

        total_precision = 0
        num_queries = len(doc_IDs_ordered)

        for i in range(num_queries):
            relevant_docs = []
            for item in qrels:
                if int(query_ids[i]) == int(item['query_num']):
                    relevant_docs.append(int(item['id']))
            precision = self.queryPrecision(doc_IDs_ordered[i], query_ids[i], relevant_docs, k)
            total_precision += precision

        mean_precision = float(total_precision / num_queries)

        return mean_precision

    def queryRecall(self, query_doc_IDs_ordered: List[int], query_id: int, true_doc_IDs: List[int], k: int) -> float:
        """
        Computation of recall of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        query_doc_IDs_ordered : List[int]
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        query_id : int
            The ID of the query in question
        true_doc_IDs : List[int]
            The list of IDs of documents relevant to the query (ground truth)
        k : int
            The k value

        Returns
        -------
        float
            The recall value as a number between 0 and 1
        """

        relevant_docs = true_doc_IDs
        retrieved_docs = query_doc_IDs_ordered[:k]

        relevant_retrieved_docs = intersection(relevant_docs, retrieved_docs)
        recall = len(relevant_retrieved_docs) / len(relevant_docs) if len(relevant_docs) != 0 else 0

        return recall

    def meanRecall(self, doc_IDs_ordered: List[List[int]], query_ids: List[int], qrels: List[dict], k: int) -> float:
        """
        Computation of recall of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        doc_IDs_ordered : List[List[int]]
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        query_ids : List[int]
            A list of IDs of the queries for which the documents are ordered
        qrels : List[dict]
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        k : int
            The k value

        Returns
        -------
        float
            The mean recall value as a number between 0 and 1
        """

        total_recall = 0
        num_queries = len(doc_IDs_ordered)

        for i in range(num_queries):
            relevant_docs = []
            for item in qrels:
                if int(query_ids[i]) == int(item['query_num']):
                    relevant_docs.append(int(item['id']))
            recall = self.queryRecall(doc_IDs_ordered[i], query_ids[i], relevant_docs, k)
            total_recall += recall

        mean_recall = total_recall / num_queries 

        return mean_recall

    def queryFscore(self, query_doc_IDs_ordered: List[int], query_id: int, true_doc_IDs: List[int], k: int) -> float:
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        query_doc_IDs_ordered : List[int]
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        query_id : int
            The ID of the query in question
        true_doc_IDs : List[int]
            The list of IDs of documents relevant to the query (ground truth)
        k : int
            The k value

        Returns
        -------
        float
            The fscore value as a number between 0 and 1
        """

        precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)

        fscore = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        return fscore

    def meanFscore(self, doc_IDs_ordered: List[List[int]], query_ids: List[int], qrels: List[dict], k: int) -> float:
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        doc_IDs_ordered : List[List[int]]
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        query_ids : List[int]
            A list of IDs of the queries for which the documents are ordered
        qrels : List[dict]
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        k : int
            The k value
        
        Returns
        -------
        float
            The mean fscore value as a number between 0 and 1
        """

        total_fscore = 0
        num_queries = len(doc_IDs_ordered)

        for i in range(num_queries):
            relevant_docs = []
            for item in qrels:
                if int(query_ids[i]) == int(item['query_num']):
                    relevant_docs.append(int(item['id']))
            fscore = self.queryFscore(doc_IDs_ordered[i], query_ids[i], relevant_docs, k)
            total_fscore += fscore

        mean_fscore = total_fscore / num_queries if num_queries != 0 else 0

        return mean_fscore

    def queryNDCG(self, query_doc_IDs_ordered: List[int], query_id: int, true_doc_IDs: List[int], k: int) -> float:
        """
        Computation of nDCG of the Information Retrieval System
        at given value of k for a single query

        Parameters
        ----------
        query_doc_IDs_ordered : List[int]
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        query_id : int
            The ID of the query in question
        true_doc_IDs : List[int]
            The list of IDs of documents relevant to the query (ground truth)
        k : int
            The k value

        Returns
        -------
        float
            The nDCG value as a number between 0 and 1
        """

        # Compute DCG
        DCG = 0
        for i in range(min(k, len(query_doc_IDs_ordered))):
            doc_id = query_doc_IDs_ordered[i]
            if doc_id in true_doc_IDs:
                relevance = 1 / (i + 1)
                DCG += relevance

        # Compute ideal DCG
        ideal_DCG = sum([1 / (i + 1) for i in range(min(k, len(true_doc_IDs)))])

        # Compute nDCG
        nDCG = DCG / ideal_DCG if ideal_DCG != 0 else 0

        return nDCG

    def meanNDCG(self, doc_IDs_ordered: List[List[int]], query_ids: List[int], qrels: List[dict], k: int) -> float:
        """
        Computation of nDCG of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        doc_IDs_ordered : List[List[int]]
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        query_ids : List[int]
            A list of IDs of the queries for which the documents are ordered
        qrels : List[dict]
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        k : int
            The k value

        Returns
        -------
        float
            The mean nDCG value as a number between 0 and 1
        """

        total_NDCG = 0
        num_queries = len(doc_IDs_ordered)

        for i in range(num_queries):
            relevant_docs = []
            for item in qrels:
                if int(query_ids[i]) == int(item['query_num']):
                    relevant_docs.append(int(item['id']))
            nDCG = self.queryNDCG(doc_IDs_ordered[i], query_ids[i], relevant_docs, k)
            total_NDCG += nDCG

        mean_NDCG = total_NDCG / num_queries if num_queries != 0 else 0

        return mean_NDCG

    def queryAveragePrecision(self, query_doc_IDs_ordered: List[int], query_id: int, true_doc_IDs: List[int], k: int) -> float:
        """
        Computation of average precision of the Information Retrieval System
        at a given value of k for a single query (the average of precision@i
        values for i such that the ith document is truly relevant)

        Parameters
        ----------
        query_doc_IDs_ordered : List[int]
            A list of integers denoting the IDs of documents in
            their predicted order of relevance to a query
        query_id : int
            The ID of the query in question
        true_doc_IDs : List[int]
            The list of documents relevant to the query (ground truth)
        k : int
            The k value

        Returns
        -------
        float
            The average precision value as a number between 0 and 1
        """

        relevant_docs = set(true_doc_IDs)
        precision_sum = 0
        num_relevant_docs_seen = 0

        for i in range(min(k, len(query_doc_IDs_ordered))):
            doc_id = query_doc_IDs_ordered[i]
            if doc_id in relevant_docs:
                num_relevant_docs_seen += 1
                precision_at_i = num_relevant_docs_seen / (i + 1)
                precision_sum += precision_at_i

        avg_precision = precision_sum / min(len(relevant_docs), k) if len(relevant_docs) != 0 else 0

        return avg_precision

    def meanAveragePrecision(self, doc_IDs_ordered: List[List[int]], query_ids: List[int], q_rels: List[dict], k: int) -> float:
        """
        Computation of MAP of the Information Retrieval System
        at given value of k, averaged over all the queries

        Parameters
        ----------
        doc_IDs_ordered : List[List[int]]
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        query_ids : List[int]
            A list of IDs of the queries
        q_rels : List[dict]
            A list of dictionaries containing document-relevance
            judgements - Refer cran_qrels.json for the structure of each
            dictionary
        k : int
            The k value

        Returns
        -------
        float
            The MAP value as a number between 0 and 1
        """

        total_avg_precision = 0
        num_queries = len(doc_IDs_ordered)

        for i in range(num_queries):
            relevant_docs = []
            for item in q_rels:
                if int(query_ids[i]) == int(item['query_num']):
                    relevant_docs.append(int(item['id']))
            avg_precision = self.queryAveragePrecision(doc_IDs_ordered[i], query_ids[i], relevant_docs, k)
            total_avg_precision += avg_precision

        mean_avg_precision = total_avg_precision / num_queries if num_queries != 0 else 0

        return mean_avg_precision

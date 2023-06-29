from ChatBot.entity import DataFormatConfig
import json
from ChatBot.logging import logger

class DataFormat:
    def __init__(self, config: DataFormatConfig):
        self.config = config
    
    # Splits each line of the file to create lines and conversations
    def loadLinesAndConversations(self):
        lines = {}
        conversations = {}
        with open(self.config.utterances, 'r', encoding='iso-8859-1') as f:
            for line in f:
                lineJson = json.loads(line)
                # Extract fields for line object
                lineObj = {}
                lineObj["lineID"] = lineJson["id"]
                lineObj["characterID"] = lineJson["speaker"]
                lineObj["text"] = lineJson["text"]
                lines[lineObj['lineID']] = lineObj

                # Extract fields for conversation object
                if lineJson["conversation_id"] not in conversations:
                    convObj = {}
                    convObj["conversationID"] = lineJson["conversation_id"]
                    convObj["movieID"] = lineJson["meta"]["movie_id"]
                    convObj["lines"] = [lineObj]
                else:
                    convObj = conversations[lineJson["conversation_id"]]
                    convObj["lines"].insert(0, lineObj)
                conversations[convObj["conversationID"]] = convObj
        logger.info("Loaded Lines and Coversations successfully!")
        return lines, conversations

    # Extracts pairs of sentences from conversations
    @staticmethod
    def extractSentencePairs(conversations):
        qa_pairs = []
        for conversation in conversations.values():
            # Iterate over all the lines of the conversation
            for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
                inputLine = conversation["lines"][i]["text"].strip()
                targetLine = conversation["lines"][i+1]["text"].strip()
                # Filter wrong samples (if one of the lists is empty)
                if inputLine and targetLine:
                    qa_pairs.append([inputLine, targetLine])
        logger.info("Extracted Sentence Pairs")
        return qa_pairs
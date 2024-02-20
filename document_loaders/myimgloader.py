from typing import List
from langchain.document_loaders.unstructured import UnstructuredFileLoader


class RapidOCRLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def img2text(filepath):
            #from rapidocr_onnxruntime import RapidOCR
            resp = ""
            #ocr = RapidOCR()

            from rapidocr_paddle import RapidOCR
            #import numpy as np
            ocr = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
            result, _ = ocr(filepath)
            if result:
                ocr_result = [line[1] for line in result]
                resp += "\n".join(ocr_result)
            return resp

        text = img2text(self.file_path)
        from unstructured.partition.text import partition_text
        return partition_text(text=text, **self.unstructured_kwargs)


if __name__ == "__main__":
    loader = RapidOCRLoader(file_path="../tests/samples/ocr_test.jpg")
    docs = loader.load()
    print(docs)

# coding=utf8

from langchain.utilities import BingSearchAPIWrapper
from langchain.docstore.document import Document


# Bing 搜索必备变量
# 使用 Bing 搜索需要使用 Bing Subscription Key,需要在azure port中申请试用bing search
# 具体申请方式请见
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/create-bing-search-service-resource
# 使用python创建bing api 搜索实例详见:
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/python
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
# 注意不是bing Webmaster Tools的api key，

# 此外，如果是在服务器上，报Failed to establish a new connection: [Errno 110] Connection timed out
# 是因为服务器加了防火墙，需要联系管理员加白名单，如果公司的服务器的话，就别想了GG
BING_SUBSCRIPTION_KEY = ""


def bing_search(text, result_len=3):
    if not (BING_SEARCH_URL and BING_SUBSCRIPTION_KEY):
        return [
            {
                "snippet": "please set BING_SUBSCRIPTION_KEY and BING_SEARCH_URL in os ENV",
                "title": "env info is not found",
                "link": "https://python.langchain.com/en/latest/modules/agents/tools/examples/bing_search.html",
            }
        ]
    search = BingSearchAPIWrapper(
        bing_subscription_key=BING_SUBSCRIPTION_KEY, bing_search_url=BING_SEARCH_URL
    )
    return search_result2docs(search.results(text, result_len))


def search_result2docs(search_results):
    docs = []
    for result in search_results:
        doc = Document(
            page_content=result["snippet"] if "snippet" in result.keys() else "",
            metadata={
                "source": result["link"] if "link" in result.keys() else "",
                "filename": result["title"] if "title" in result.keys() else "",
            },
        )
        docs.append(doc)
    return docs


if __name__ == "__main__":
    r = bing_search("python")
    print(r)

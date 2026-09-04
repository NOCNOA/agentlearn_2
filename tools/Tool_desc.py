TOOLS = [
    {
        "type": "function",

        "function": {

            "name": "search_arxiv",

            "description": (
                "Search the online paper research tool using a keyword. "
                "Use this tool when the user asks about papers, "
                "research methods, or related literature."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {
                        "type": "string",
                        "description": (
                            "The keyword used to search papers, "
                            "for example: large disparity"
                        ),
                    },
                    
                    "max_results": {
                        "type": "int",
                        "description": (
                            "The number that papers return, "
                            "for example: 5"
                        ),
                    },
                },

                "required": ["query", "max_results"],

            },
        },
    },
        {
        "type": "function",
        "function": {
            "name": "get_arxiv_paper",
            "description": (
                "根据 arXiv ID 读取一篇论文的详细信息，"
                "包括完整摘要、作者、发布时间和论文类别。"
                "应先使用 search_arxiv 找到论文 ID，"
                "再用本工具读取最相关论文。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": (
                            "论文的 arXiv ID，例如 2608.20788；"
                            "也可以传入 arXiv URL。"
                        ),
                    }
                },
                "required": ["arxiv_id"],
                "additionalProperties": False,
            },
        },
    }
]



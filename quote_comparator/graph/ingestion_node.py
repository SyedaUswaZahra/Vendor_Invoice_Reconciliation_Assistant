parsed_quotes=state.get("parsed_quotes", processed_keys=state.get("processed_source_keys", raw_inputs=state.get("raw_inputs", source_type=source_type, key=raw.get("key", raw_text=raw_text, doc=QuoteDocument(, loader=PyPDFLoader(path, pages=loader.load(, llm=ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0, image_data=base64.b64encode(f.read(, message=HumanMessage(, content=[
                    {
                        "type": "text",
                        "text": (
                            "Please transcribe the construction estimate or quote shown in this image "
                            "into plain text. Include all line items, descriptions, quantities, unit "
                            "prices, and totals exactly as they appear."
                        ),
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                ], response=llm.invoke([message], text=pytesseract.image_to_string(Image.open(path
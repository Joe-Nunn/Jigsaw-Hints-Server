#!/bin/sh

curl -X POST $1 -H "Content-Type: application/json" --data-binary "@request.json"

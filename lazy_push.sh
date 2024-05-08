#!/bin/sh

if [ -z "$(git status --porcelain)" ];
then 
    echo "Working directory clean"
else 
    echo "Uncommitted changes"
    date=`date`
    git add .
    git commit -m "Lazy push: $date"
    git push
fi
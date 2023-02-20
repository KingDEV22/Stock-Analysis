import git
import subprocess
import datetime
repo = git.Repo('.')
# Provide a commit message

subprocess.check_output("git add .", stderr=subprocess.PIPE)
repo.index.commit('news fetch for' + str(datetime.date.today()))

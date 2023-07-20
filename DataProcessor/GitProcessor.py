import git
from shutil import copyfile

def getRepo(path):
    def _reset_repo(repo):
        if checkout(repo, 'master'):
            print('checkout master')
            return True
        elif checkout(repo, 'main'):
            print('checkout main')
            return True
        elif checkout(repo, 'origin'):
            print('checkout origin')
            return True
        return False
    repo = git.Repo(path)
    assert not repo.bare

    if _reset_repo(repo):
        print("\n=======  reset Success =======")
    else:
        print("\n========  reset Faile  ========")
    return repo
    
def checkout(repo, commit, force=True):
    try:
        repo.git.checkout(commit, force=force)
        print("checkout success")
        return True
    except Exception as e:
        print("checkout failed")
        return False

def copy_sc(repo, bug_reports, save_base_path):
    project_path = repo.common_dir[:-4]

    for bug_report in bug_reports:
        bug_report.setNewFiles()
        commit = bug_report.getBuggyCommit()
        checkout(repo, commit)

        files = bug_report.getFiles()

        not_exist_files = []
        for file_path in files:
            f_path = project_path + file_path
            s_path = save_base_path + commit +"_"+ file_path.replace("/", "_")
            try:
                _copyFile(f_path, s_path)
                bug_report.addNewFile(s_path)
            except:
                not_exist_files.append(file_path)
        
        commit = bug_report.getFixedCommit()
        checkout(repo, commit)
        for file_path in not_exist_files:
            f_path = project_path + file_path
            s_path = save_base_path + commit +"_"+ file_path.replace("/", "_")
            _copyFile(f_path, s_path)
            bug_report.addNewFile(s_path)

def _copyFile(file_path, save_path):
    copyfile(file_path, save_path)

def getAllCommits(repo):
    commits = []
    for commit in repo.iter_commits():
        commits.append( (commit.committed_date, commit ) )

    commits = [commit for _, commit in sorted(commits, key=lambda x: x[0])]
    return commits
import os
import send2trash

######################################################################
def gauge(ii, total):
  ###
  percentage = (ii + 1) / total  # 現在の進行状況を計算
  bar_length = 20  # ゲージの全長
  progress = int(bar_length * percentage)  # 完了部分の長さ
  bar = '[' + '=' * progress + '.' * (bar_length - progress) + ']'  # ゲージの形成
  print(f'\r{bar} {percentage * 100:.2f}%', end='')  # ゲージと進行状況のパーセンテージを表示

import shutil
######################################################################
def print_line():
    # ターミナルの幅を取得
    width = shutil.get_terminal_size().columns
    # 幅いっぱいに '-' で埋めた文字列を出力
    print('-' * width)

######################################################################
def compress_ranges(nums):
    if isinstance(nums, int):  # If a single integer is passed
        return [str(nums)]
    
    nums = nums.tolist()

    compressed = []
    start = nums[0]
    end = nums[0]

    for i in range(1, len(nums)):
        if nums[i] == end + 1:
            end = nums[i]
        else:
            if start == end:
                compressed.append(str(start))
            else:
                compressed.append(f"{start}:{end}")
            start = nums[i]
            end = nums[i]

    if start == end:
        compressed.append(str(start))
    else:
        compressed.append(f"{start}:{end}")

    return compressed

######################################################################
def safe_delete_folder(directory):
    send2trash(directory)
    print(f"Sent {directory} to the trash as it contains no results.")

######################################################################
def safe_delete_folders_without_ext(directory, ext):
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            contains_png = False
            for file in os.listdir(subdir_path):
                if file.endswith(ext):
                    contains_png = True
                    break
                if file.endswith('.mp4'):
                    contains_png = True
                    break
            if not contains_png:
                send2trash.send2trash(subdir_path)
                print(f"Sent {subdir_path} to the trash as it contains no .png and .mp4 files.")

######################################################################
import os
from send2trash import send2trash

def delete_folders(directory, exts, exceptions=None):

    black, blue, yellow, red, b_black, b_blue, b_yellow, b_red, reset = create_color_functions()

    if exceptions is None:
        exceptions = []

    # 拡張子を小文字に統一
    exts = [e.lower() for e in exts]

    try:
        # メインディレクトリの直下にあるエントリをリストアップ
        for entry in os.listdir(directory):
            entry_path = os.path.join(directory, entry)

            # サブディレクトリのみを対象
            if not os.path.isdir(entry_path):
                continue

            # 例外リストに含まれる場合はスキップ
            if entry in exceptions:
                print(f"{b_blue}Skip EXCEPTION{reset}: {os.path.relpath(entry_path, directory)}")
                continue

            # 指定拡張子のファイルが含まれているか再帰的にチェック
            contains_desired_ext = False
            for root, dirs, files in os.walk(entry_path):
                # 例外フォルダ内のファイルも検査対象にしたい場合は、以下のコメントを外してください
                # if os.path.basename(root) in exceptions:
                #     dirs[:] = []  # そのサブディレクトリ内の更なるサブディレクトリを無視
                for file in files:
                    _, file_ext = os.path.splitext(file)
                    if file_ext.lower() in exts:
                        contains_desired_ext = True
                        break
                if contains_desired_ext:
                    break

            if not contains_desired_ext:
                try:
                    send2trash(entry_path)
                    print(f"{b_yellow}Moved to trash{reset}: {os.path.relpath(entry_path, directory)}")
                except Exception as e:
                    print(f"{b_red}Unable to move{reset}: {os.path.relpath(entry_path, directory)}: {e}")

    except FileNotFoundError:
        print(f"Folder not found: {directory}")
    except PermissionError:
        print(f"Folder access denied: {directory}")
    except Exception as e:
        print(f"Unknown error: {e}")

# Example usage:
# delete_folders("/path/to/directory", ['.mp4', '.pkl'], exception=['ImportantFolder', 'DoNotDelete'])



######################################################################
def print_nwb_structure(group, indent=""):
    """Recursive function to print NWB structure."""
    # Handle top-level NWBFile object differently from nested groups
    if isinstance(group, dict):
        keys = list(group.keys())
    else:
        keys = [attr for attr in dir(group) if not attr.startswith('_') and not callable(getattr(group, attr))]

    for i, key in enumerate(keys):
        item = getattr(group, key) if not isinstance(group, dict) else group[key]
        
        if hasattr(item, 'keys') or isinstance(item, dict):
            print(f"{indent}┃")
            print(f"{indent}┠━ {key}")
            print_nwb_structure(item, indent + "┃   ")
        else:
            connector = "┗━" if i == len(keys) - 1 else "┠━"
            print(f"{indent}{connector} {key}")

######################################################################
# ANSI escape sequences for text colors and styles
def create_color_functions():
    # Regular colors
    black = '\033[1m'
    blue = '\033[94m'
    yellow = '\033[33m'
    red = '\033[31m'
    
    # Bold colors
    b_black = '\033[1m'
    b_blue = '\033[1;94m'
    b_yellow = '\033[1;33m'
    b_red = '\033[1;31m'
    
    # Reset color
    reset = '\033[0m'
    
    return black, blue, yellow, red, b_black, b_blue, b_yellow, b_red, reset



def print_color2():
    """ANSI escape sequences for text colors and styles"""

    # if selecting RGB color
    # '\033[38;2;R;G;Bm'

    # if selecting ANSI color
    # \033[<color>m
    # \033[1;<color>m
    
    # Regular colors
    colors = {
        "black": '\033[1m',
        "red": '\033[91m',
        "green": '\033[92m',
        "yellow": '\033[93m',
        "blue": '\033[94m',
        "magenta": '\033[95m',
        "cyan": '\033[96m',
        "white": '\033[97m',

        # Bold colors
        "b_black": '\033[1;1m',
        "b_red": '\033[1;91m',
        "b_green": '\033[1;92m',
        "b_yellow": '\033[1;93m',
        "b_blue": '\033[1;94m',
        "b_magenta": '\033[1;95m',
        "b_cyan": '\033[1;96m',
        "b_white": '\033[1;97m',

        # Highlighted colors
        "h_black": '\033[1;100m',
        "h_red": '\033[1;101m',
        "h_green": '\033[1;102m',
        "h_yellow": '\033[1;103m',
        "h_blue": '\033[1;104m',
        "h_magenta": '\033[1;105m',
        "h_cyan": '\033[1;106m',
        "h_white": '\033[1;107m',

        "h_orange": '\033[1;48;2;51;255;153m',
        
        # Reset color
        "reset": '\033[0m'
    }
    
    return colors

######################################################################
def print_section(Text, type="Main"):
    black, blue, yellow, red, b_black, b_blue, b_yellow, b_red, reset = create_color_functions()
    if type == "Main":
        print("\n" + (f"{b_black}{Text}{reset}").center(150, '*'))
    elif type == "Sub":
        print("\n" + (f"{b_black}{Text}{reset}").center(50, '#'))

######################################################################
def get_brain_abbr(chan_name):
    # 領域名と対応する略称の辞書を定義
    brain_regions = {
        #'caudalmiddlefrontal': 'cMFG',
        'caudalmiddlefrontal': 'MFG',
        #'parsopercularis': 'POp', # 狭義の Broca 野
        'parsopercularis': 'IFG',
        
        'precentral': 'PrCG',
        'postcentral': 'PoCG',
        'supramarginal': 'SMG',
        'superiortemporal': 'STG',
        'middletemporal': 'MTG',

        'inferiorfrontal': 'IFG'

        # for TJ
        # 'J': 
        # 'L':
        # 'S':
        # 'U':
        # 'W':

    }

    # 入力された領域名に対応する略称を返す(なかったらそのまま)
    return brain_regions.get(chan_name, chan_name)

######################################################################
def RenewFileStamp(DataFolder):
    """
    DataFolder: str
        Data folder name
    """
    # 現在の時間を取得
    from datetime import datetime
    import pytz

    # 現在の時間を取得
    cst = pytz.timezone('US/Central') # Current Location Time: CST
    current_time = datetime.now(cst).strftime('%Y%m%d_%H%M')

    import re
    m = re.search(r'(\d{8}_\d{4})_(.+)$', DataFolder)
    if m:
        old_time, name_part = m.groups()
        SaveName = f"{current_time}_{name_part}"  # 古いタイムスタンプを新しいものに置き換え
    else:
        SaveName = f"{current_time}_{DataFolder}"  # パターンにマッチしない場合は元の処理

    return SaveName

def GetFolderName(TargetDir):
    
    DataFolder = input(f"-- Enter a dataset folder name: ") # Data folder name input
    if DataFolder == "":
        # DataFolder = 'features prp'  # Default name
        folders_list = [f for f in os.listdir(TargetDir) 
                        if os.path.isdir(os.path.join(TargetDir, f)) and f != 'Results']
        DataFolder = max(folders_list, key=lambda f: os.path.getctime(os.path.join(TargetDir, f))) # the most recent folder
        print(f"-- The most recent folder name to be loaded: ./\033[1m{DataFolder}/\033[0m")
    else:
        print(f"-- Data folder name to be loaded: ./\033[1m{DataFolder}/\033[0m")

    return DataFolder

def select_folder(title="Select Folder", initialdir="", allow_cli_fallback=True):
    """
    フォルダ選択ダイアログ（macOSでの安定動作を優先）。

    優先度:
      1) macOS + PyObjC(Apple公式API) が使える場合は NSOpenPanel を使用
      2) Tkinter の askdirectory を使用（最前面/センタリング対策込み）
      3) GUI が使えない/失敗時は CLI 入力でフォルダを取得（キャンセル可）

    Parameters:
    -----------
    title : str
        ダイアログウィンドウのタイトル
    initialdir : str
        初期ディレクトリのパス
    allow_cli_fallback : bool
        GUI に失敗した場合に CLI 入力で代替するかどうか

    Returns:
    --------
    str or None
        選択されたフォルダのパス。キャンセル/失敗時は None
    """
    import os
    import platform

    # 初期ディレクトリの正規化
    initialdir = os.path.expanduser(initialdir or "~")
    if not os.path.isdir(initialdir):
        try:
            os.makedirs(initialdir, exist_ok=True)
        except Exception:
            initialdir = os.path.expanduser("~")

    # 1) macOS + PyObjC (NSOpenPanel)
    if platform.system() == "Darwin":
        try:
            import importlib
            AppKit = importlib.import_module("AppKit")
            Foundation = importlib.import_module("Foundation")
            NSOpenPanel = getattr(AppKit, "NSOpenPanel")
            NSModalResponseOK = getattr(AppKit, "NSModalResponseOK", 1)
            NSURL = getattr(Foundation, "NSURL")

            panel = NSOpenPanel.openPanel()
            panel.setCanChooseFiles_(False)
            panel.setCanChooseDirectories_(True)
            panel.setAllowsMultipleSelection_(False)
            panel.setPrompt_("Select")
            panel.setTitle_(title)
            # 初期ディレクトリ
            url = NSURL.fileURLWithPath_(initialdir)
            panel.setDirectoryURL_(url)

            # NSOpenPanel を実行
            result = panel.runModal()
            if result == NSModalResponseOK:
                urls = panel.URLs()
                if urls and len(urls) > 0:
                    return str(urls[0].path())
            # キャンセルなど
        except Exception:
            # PyObjC が無い/動かない → Tkinter にフォールバック
            pass

    # 2) Tkinter askdirectory
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()  # メインウィンドウ非表示
        # 最前面に
        try:
            root.attributes('-topmost', True)
        except Exception:
            pass
        # ウィンドウ位置（センタリング）
        try:
            root.update_idletasks()
            root.call('tk', 'scaling', 2.0)  # HiDPI の場合の文字サイズ調整（無害）
            root.call('tk::PlaceWindow', root._w, 'center')
        except Exception:
            pass

        folder_path = filedialog.askdirectory(title=title, initialdir=initialdir, mustexist=False)
        try:
            root.destroy()
        except Exception:
            pass

        if folder_path:
            return folder_path
    except Exception:
        # Tk が無い/使えない環境
        pass

    # 3) CLI フォールバック
    if allow_cli_fallback:
        try:
            black, blue, yellow, red, b_black, b_blue, b_yellow, b_red, reset = create_color_functions()
            print(f"{b_yellow}[select_folder]{reset} GUI が使用できないため、CLI でフォルダを入力してください。")
            print(f" 初期ディレクトリの候補: {initialdir}")
            path = input("Folder path (空Enterでキャンセル): ").strip()
            if not path:
                return None
            path = os.path.expanduser(path)
            if os.path.isdir(path):
                return path
            else:
                try:
                    os.makedirs(path, exist_ok=True)
                    return path
                except Exception as e:
                    print(f"{b_red}無効なパス/作成不可{reset}: {e}")
                    return None
        except Exception:
            return None

    return None
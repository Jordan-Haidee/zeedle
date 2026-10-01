# Song Item Context Menu Design

## Goal

Add right-click actions to song rows in both the Gallery list and Search results:

- Delete the selected song and its audio file from disk.
- Edit the song title and artist through a popup with prefilled text inputs.

## UI

Use Slint's built-in `ContextMenuArea` on each `SongItem` so right-click and the keyboard menu key open the same accessible menu. The menu contains Edit metadata and Delete actions. Both `SongListView` and `SearchPanel` forward the selected `SongInfo` through callbacks to the app root.

The metadata popup is an app-level dialog with title and artist `LineEdit` fields, plus Save and Cancel. Save is disabled or ignored when both values are blank. Cancel and clicking outside close the popup without changing the file. The dialog reports file-write errors in the dialog rather than closing as though the save succeeded.

## Data and file operations

Use the selected song's path as its stable identity. A metadata-save request writes title and artist to the audio file tags through Lofty. Only after a successful write does the UI replace that song's metadata, update the current-song display if applicable, and recompute search results from the full song list. The playback stream remains open because the path is unchanged.

Delete removes the audio file first. If removal fails, leave the list, search results, and playback state unchanged and show the error. After successful removal, remove the song from the source list and recompute search results. If it was the current song, stop the old stream and play the next available song; if no songs remain, clear the player and current-song state. If another song was playing, keep it playing.

## Boundaries

Do not add a separate soft-delete database or change startup scanning. Files removed from disk disappear naturally on subsequent scans. Preserve existing playback, sort, and search behavior for songs that are not edited or deleted.

## Verification

Compile the Slint UI and Rust application. Exercise menu opening and both actions in Gallery and Search. Verify tag changes survive a restart, file deletion removes the file and updates both views, playback transitions correctly when deleting the current song, and failed file operations do not mutate the UI state. Review a rendered app view of the popup and context menu.

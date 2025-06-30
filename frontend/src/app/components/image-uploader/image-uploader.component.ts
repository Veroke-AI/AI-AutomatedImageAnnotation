import { Component, EventEmitter, Output, NgZone } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';

@Component({
  selector: 'app-image-uploader',
  standalone: true,
  imports: [
    CommonModule,
    MatIconModule,
    MatButtonModule
  ],
  template: `
    <div
      class="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-primary-500 transition-colors"
      [class.border-primary-500]="isDragging"
      (dragover)="onDragOver($event)"
      (dragleave)="onDragLeave($event)"
      (drop)="onDrop($event)"
    >
      <input
        #folderInput
        type="file"
        webkitdirectory
        multiple
        class="hidden"
        (change)="onFolderSelected($event)"
      />
      
      <mat-icon class="text-4xl text-gray-400 mb-2" style="height: 35px; width: 35px;">folder_open</mat-icon>
      <p class="text-gray-600 mb-2">Drag & drop a folder here</p>
      <p class="text-gray-500 text-sm mb-4">or</p>
      <button
        mat-raised-button
        color="primary"
        (click)="folderInput.click()"
      >
        Select Folder
      </button>

      <!--<div *ngIf="uploadedFiles.length > 0" class="mt-4 text-left">
        <p class="text-sm text-gray-600 mb-2">
          {{ uploadedFiles.length }} images selected
        </p>
        <ul class="text-xs text-gray-500 space-y-1 max-h-32 overflow-y-auto">
          <li *ngFor="let file of uploadedFiles">
            {{ file.name }}
          </li>
        </ul>
      </div>-->
    </div>
  `,
  styles: []
})
export class ImageUploaderComponent {
  @Output() imagesUploaded = new EventEmitter<File[]>();
  
  uploadedFiles: File[] = [];
  isDragging = false;

  constructor(private ngZone: NgZone) {}

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = true;
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = false;
  }

  onDrop(event: DragEvent): void {
    console.log('Drop event triggered');
    event.preventDefault();
    event.stopPropagation();
    this.isDragging = false;

    const items = event.dataTransfer?.items;
    if (!items) {
      console.log('No items in drop event');
      return;
    }

    // Clear previous files
    this.uploadedFiles = [];
    const imageFiles: File[] = [];
    let pendingEntries = 0;

    const emitFiles = () => {
      console.log('Emitting files:', imageFiles.length);
      if (imageFiles.length > 0) {
        const sortedFiles = [...imageFiles].sort((a, b) => a.name.localeCompare(b.name));
        this.ngZone.run(() => {
          this.uploadedFiles = sortedFiles;
          this.imagesUploaded.emit(sortedFiles);
        });
      }
    };

    const processEntry = (entry: FileSystemEntry | null) => {
      if (!entry) return;

      if (entry.isFile) {
        pendingEntries++;
        (entry as FileSystemFileEntry).file((file: File) => {
          console.log('Processing file:', file.name);
          if (file.type.startsWith('image/')) {
            imageFiles.push(file);
            // Emit current batch of files
            emitFiles();
          }
          pendingEntries--;
          if (pendingEntries === 0) {
            console.log('All files processed');
            emitFiles();
          }
        });
      } else if (entry.isDirectory) {
        console.log('Processing directory:', entry.name);
        pendingEntries++;
        const reader = (entry as FileSystemDirectoryEntry).createReader();
        
        const readEntries = () => {
          reader.readEntries((entries) => {
            if (entries.length > 0) {
              console.log('Found entries:', entries.length);
              entries.forEach(processEntry);
              readEntries(); // Continue reading
            } else {
              pendingEntries--;
              if (pendingEntries === 0) {
                console.log('Directory processing complete');
                emitFiles();
              }
            }
          });
        };
        
        readEntries();
      }
    };

    // Process all dropped items
    Array.from(items).forEach(item => {
      const entry = item.webkitGetAsEntry();
      if (entry) {
        processEntry(entry);
      }
    });
  }

  onFolderSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files) return;

    const imageFiles = Array.from(input.files).filter(file => file.type.startsWith('image/'));
    if (imageFiles.length > 0) {
      const sortedFiles = imageFiles.sort((a, b) => a.name.localeCompare(b.name));
      this.uploadedFiles = sortedFiles;
      this.imagesUploaded.emit(sortedFiles);
    }
  }
} 
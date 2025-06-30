import { Component, Input, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTableModule } from '@angular/material/table';
import { MatSnackBar } from '@angular/material/snack-bar';
import { MatSliderModule } from '@angular/material/slider';
import { config } from '../../../../../config';

interface DuplicateGroup {
  groupId: number;
  count: number;
  sampleImages: string[];
}

@Component({
  selector: 'app-duplicate-detection',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatTableModule,
    MatSliderModule
  ],
  template: `
    <div class="p-6">
      <!-- Debug info -->
      <div *ngIf="duplicateGroups.length > 0" class="mb-4 p-4 bg-gray-100 rounded">
        <p>Number of groups: {{ duplicateGroups.length }}</p>
      </div>

      <div class="mb-6">
        <h2 class="text-lg font-medium text-gray-900 mb-2">Duplicate Image Detection</h2>
        <p class="text-sm text-gray-600">Find similar or duplicate images in your image collection</p>
      </div>

      <!-- Threshold Control -->
      <div class="mb-6">
        <div class="flex flex-col gap-2">
          <label class="text-sm font-medium text-gray-700">Similarity Threshold</label>
          <div class="flex items-center gap-4">
            <mat-slider
              class="flex-grow"
              [min]="0"
              [max]="1"
              [step]="0.01"
              [discrete]="true"
              [showTickMarks]="true"
            >
              <input 
                matSliderThumb
                [(ngModel)]="similarityThreshold"
              >
            </mat-slider>
            <div class="w-16 text-center">
              {{ similarityThreshold | number:'1.2-2' }}
            </div>
          </div>
          <p class="text-xs text-gray-500 mt-1">Higher value = more exact matches</p>
          
          <div class="mt-4">
            <button 
              mat-raised-button
              color="primary"
              [disabled]="isLoading"
              (click)="findDuplicates()"
            >
              <mat-icon class="mr-2">search</mat-icon>
              Find Duplicates
            </button>
          </div>
        </div>
      </div>

      <!-- Results Table -->
      <div *ngIf="duplicateGroups.length > 0" class="space-y-4">
        <h3 class="text-lg font-medium text-gray-900">Duplicate Groups ({{ duplicateGroups.length }})</h3>
        
        <div class="bg-white rounded-lg shadow overflow-hidden">
          <table class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
              <tr>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Group</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Count</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sample Images</th>
              </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
              <tr *ngFor="let group of duplicateGroups" 
                  class="hover:bg-gray-50 cursor-pointer"
                  (click)="showGroupImages(group)">
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{{ group.groupId }}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{{ group.count }}</td>
                <td class="px-6 py-4 text-sm text-gray-500 max-w-lg truncate">{{ group.sampleImages.join(', ') }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- No Results Message -->
      <div *ngIf="hasSearched && duplicateGroups.length === 0" class="text-center py-8">
        <mat-icon class="text-gray-400" style="font-size: 48px; width: 48px; height: 48px;">find_in_page</mat-icon>
        <p class="mt-2 text-gray-600">No duplicate images found</p>
      </div>

      <!-- Image Preview Dialog -->
      <div *ngIf="selectedGroup" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div class="bg-white rounded-lg p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
          <div class="flex justify-between items-center mb-4">
            <h3 class="text-lg font-medium text-gray-900">Group {{ selectedGroup.groupId }}</h3>
            <button mat-icon-button (click)="selectedGroup = null">
              <mat-icon>close</mat-icon>
            </button>
          </div>
          
          <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div *ngFor="let imagePath of selectedGroup.sampleImages; let i = index" 
                 class="aspect-square relative group">
              <img 
                [src]="getImagePreviewUrl(imagePath)"
                class="w-full h-full object-cover rounded-lg"
                [alt]="imagePath"
              >
              <!-- Delete button - show for all except first image -->
              <button *ngIf="i > 0"
                      mat-mini-fab
                      color="warn"
                      class="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
                      matTooltip="Delete this duplicate"
                      (click)="deleteImage(selectedGroup.groupId, imagePath, i); $event.stopPropagation()">
                <mat-icon>delete</mat-icon>
              </button>
              <!-- Show 'Original' badge on first image -->
              <div *ngIf="i === 0" 
                   class="absolute top-2 left-2 bg-green-500 text-white px-2 py-1 rounded text-sm">
                Original
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Loading Overlay -->
      <div *ngIf="isLoading" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div class="bg-white rounded-lg p-8 flex flex-col items-center">
          <div class="loader-ring">
            <div></div>
            <div></div>
            <div></div>
            <div></div>
          </div>
          <p class="mt-4 text-gray-700 font-medium">Processing...</p>
        </div>
      </div>
    </div>
  `,
  styles: [`
    :host {
      display: block;
    }

    mat-slider {
      min-width: 300px;
    }

    .loader-ring {
      display: inline-block;
      position: relative;
      width: 80px;
      height: 80px;
    }
    .loader-ring div {
      box-sizing: border-box;
      display: block;
      position: absolute;
      width: 64px;
      height: 64px;
      margin: 8px;
      border: 8px solid #3B82F6;
      border-radius: 50%;
      animation: loader-ring 1.2s cubic-bezier(0.5, 0, 0.5, 1) infinite;
      border-color: #3B82F6 transparent transparent transparent;
    }
    .loader-ring div:nth-child(1) {
      animation-delay: -0.45s;
    }
    .loader-ring div:nth-child(2) {
      animation-delay: -0.3s;
    }
    .loader-ring div:nth-child(3) {
      animation-delay: -0.15s;
    }
    @keyframes loader-ring {
      0% {
        transform: rotate(0deg);
      }
      100% {
        transform: rotate(360deg);
      }
    }
  `]
})
export class DuplicateDetectionComponent implements OnDestroy {
  @Input() uploadedFiles: File[] = [];
  similarityThreshold = 0.95;
  duplicateGroups: DuplicateGroup[] = [];
  isLoading = false;
  hasSearched = false;
  selectedGroup: DuplicateGroup | null = null;
  private previewUrls = new Map<string, string>();

  constructor(
    private http: HttpClient,
    private snackBar: MatSnackBar
  ) {}

  getImagePreviewUrl(filename: string): string {
    // Check if we already have a preview URL for this file
    if (this.previewUrls.has(filename)) {
      return this.previewUrls.get(filename)!;
    }

    // Find the matching file in uploadedFiles
    const file = this.uploadedFiles.find(f => f.name === filename.split('/').pop());
    if (file) {
      const url = URL.createObjectURL(file);
      this.previewUrls.set(filename, url);
      return url;
    }

    return ''; // Return empty string if file not found
  }

  async findDuplicates(): Promise<void> {
    if (!this.uploadedFiles?.length) {
      this.snackBar.open('Please upload images first', 'Close', {
        duration: 3000
      });
      return;
    }

    const formData = new FormData();
    formData.append('similarity_threshold', this.similarityThreshold.toString());
    
    // Add all uploaded images to formData
    this.uploadedFiles.forEach(file => {
      formData.append('images', file);
    });

    try {
      this.isLoading = true;
      const response = await this.http.post<any>(`http://${config.IP}:${config.PORT}/find_duplicates`, formData).toPromise();
      console.log('API Response:', response);

      // Transform the response into DuplicateGroup format
      this.duplicateGroups = Object.entries(response).map(([groupId, group]: [string, any]) => ({
        groupId: parseInt(groupId.replace('group_', '')),
        count: group.length,
        sampleImages: group
      }));

      console.log('Transformed duplicateGroups:', this.duplicateGroups);
      this.hasSearched = true;

      if (this.duplicateGroups.length === 0) {
        this.snackBar.open('No duplicate images found', 'Close', {
          duration: 3000
        });
      }
    } catch (error) {
      console.error('Error finding duplicates:', error);
      this.snackBar.open('Error finding duplicates', 'Close', {
        duration: 3000
      });
    } finally {
      this.isLoading = false;
    }
  }

  showGroupImages(group: DuplicateGroup): void {
    this.selectedGroup = group;
  }

  async deleteImage(groupId: number, imagePath: string, index: number): Promise<void> {
    try {
      // Show confirmation dialog
      if (!confirm(`Are you sure you want to delete this duplicate image?\n${imagePath}`)) {
        return;
      }

      this.isLoading = true;

      // Get the file name from the path
      const fileName = imagePath.split('/').pop();
      if (!fileName) {
        throw new Error('Invalid file path');
      }

      // Remove the file from uploadedFiles array
      const fileIndex = this.uploadedFiles.findIndex(file => file.name === fileName);
      if (fileIndex !== -1) {
        this.uploadedFiles.splice(fileIndex, 1);
      }

      // Update both the selected group and main groups array
      const groupIndex = this.duplicateGroups.findIndex(g => g.groupId === groupId);
      if (groupIndex !== -1) {
        // Update the group's image list
        this.duplicateGroups[groupIndex].sampleImages = this.duplicateGroups[groupIndex].sampleImages
          .filter(path => path !== imagePath);
        this.duplicateGroups[groupIndex].count--;

        // If this is the currently selected group, update it as well
        if (this.selectedGroup && this.selectedGroup.groupId === groupId) {
          this.selectedGroup = { ...this.duplicateGroups[groupIndex] };
        }

        // Remove the group only if exactly one image remains
        if (this.duplicateGroups[groupIndex].count === 1) {
          this.duplicateGroups = this.duplicateGroups.filter(g => g.groupId !== groupId);
          // Close the dialog if we're removing the current group
          if (this.selectedGroup && this.selectedGroup.groupId === groupId) {
            this.selectedGroup = null;
          }
          this.snackBar.open('Group removed - only one image remained', 'Close', { duration: 3000 });
        }
      }

      // Revoke the URL for the deleted image
      const previewUrl = this.previewUrls.get(imagePath);
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        this.previewUrls.delete(imagePath);
      }

      this.snackBar.open('Duplicate image removed successfully', 'Close', { duration: 3000 });
    } catch (error) {
      console.error('Error removing image:', error);
      this.snackBar.open('Error removing image', 'Close', { duration: 3000 });
    } finally {
      this.isLoading = false;
    }
  }

  ngOnDestroy(): void {
    // Clean up preview URLs
    this.previewUrls.forEach(url => URL.revokeObjectURL(url));
    this.previewUrls.clear();
  }
} 
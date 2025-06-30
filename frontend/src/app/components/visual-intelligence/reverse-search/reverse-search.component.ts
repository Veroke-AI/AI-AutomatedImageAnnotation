import { Component, Input, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSnackBar } from '@angular/material/snack-bar';
import { HttpClient } from '@angular/common/http';
import { config } from '../../../../../config';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';

interface SearchResult {
  image_index: number;
  filename: string;
  similarity_score: number;
  status: string;
}

interface SearchResponse {
  reference_image: string;
  total_images: number;
  processed_images: number;
  results: SearchResult[];
}

@Component({
  selector: 'app-reverse-search',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatButtonModule,
    MatIconModule
  ],
  template: `
    <div class="p-6">
      <div class="mb-6">
        <h2 class="text-lg font-medium text-gray-900 mb-2">Reverse Image Search</h2>
        <p class="text-sm text-gray-600">Find similar images by using a reference image</p>
      </div>

      <!-- Image Upload -->
      <div class="mb-6">
        <div class="flex items-center gap-4">
          <div class="flex-grow">
            <input 
              #fileInput
              type="file"
              (change)="onFileSelected($event)"
              accept="image/*"
              class="hidden"
            >
            <input 
              type="text"
              [value]="selectedFile?.name || ''"
              readonly
              class="w-full px-4 py-2 border rounded-lg bg-gray-50"
              placeholder="g/Uni Fellows/Trail 5/download.jpg"
            >
          </div>
          <button 
            mat-raised-button
            color="primary"
            (click)="fileInput.click()"
          >
            Browse...
          </button>
        </div>

        <!-- Preview -->
        <div *ngIf="selectedFile" class="mt-4 flex justify-center">
          <div class="relative w-48 h-48">
            <img 
              [src]="previewUrl"
              class="w-full h-full object-cover rounded-lg shadow-sm"
              alt="Reference image"
            >
          </div>
        </div>

        <!-- Search Button -->
        <div class="mt-4 flex justify-center">
          <button
            mat-raised-button
            color="primary"
            [disabled]="!selectedFile || isLoading || !uploadedFiles?.length"
            (click)="search()"
          >
            <mat-icon>search</mat-icon>
            Find Similar Images
          </button>
        </div>
      </div>

      <!-- Results Grid -->
      <div *ngIf="searchResults.length > 0" class="space-y-4">
        <div class="flex justify-between items-center">
          <h3 class="text-lg font-medium text-gray-900">Search Results</h3>
          <div class="flex gap-2">
            <button 
              mat-raised-button
              color="primary"
              (click)="downloadResults()"
            >
              <mat-icon>download</mat-icon>
              Download Results
            </button>
            <button 
              mat-raised-button
              color="primary"
              (click)="createSlideshow()"
            >
              <mat-icon>slideshow</mat-icon>
              Create Slideshow
            </button>
          </div>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          <div *ngFor="let result of searchResults" class="relative group">
            <img 
              [src]="getImagePreviewUrl(result.filename)"
              class="w-full aspect-square object-cover rounded-lg shadow-sm"
              [alt]="result.filename"
            >
            <div class="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white p-2 text-sm">
              <div class="truncate">{{ result.filename }}</div>
              <div>Similarity: {{ result.similarity_score.toFixed(4) }}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- No Results Message -->
      <div *ngIf="hasSearched && searchResults.length === 0" class="text-center py-8">
        <mat-icon class="text-gray-400" style="font-size: 48px; width: 48px; height: 48px;">search_off</mat-icon>
        <p class="mt-2 text-gray-600">No similar images found</p>
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
export class ReverseSearchComponent implements OnDestroy {
  @Input() uploadedFiles: File[] = [];
  selectedFile: File | null = null;
  previewUrl: string | null = null;
  searchResults: SearchResult[] = [];
  isLoading = false;
  hasSearched = false;
  protected readonly config = config;
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

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      this.selectedFile = input.files[0];
      
      // Create preview URL
      const reader = new FileReader();
      reader.onload = (e) => {
        this.previewUrl = e.target?.result as string;
      };
      reader.readAsDataURL(this.selectedFile);
    }
  }

  async search(): Promise<void> {
    if (!this.selectedFile || !this.uploadedFiles?.length) {
      console.log('Validation failed:', {
        hasSelectedFile: !!this.selectedFile,
        uploadedFilesLength: this.uploadedFiles?.length
      });
      return;
    }

    const formData = new FormData();
    formData.append('reference_image', this.selectedFile);
    
    // Add all uploaded images to formData
    this.uploadedFiles.forEach(file => {
      formData.append('images', file);
    });

    try {
      this.isLoading = true;
      const response = await this.http.post<SearchResponse>(`http://${config.IP}:${config.PORT}/reverse_search`, formData).toPromise();

      if (response) {
        this.searchResults = response.results;
        this.hasSearched = true;

        if (!response.results?.length) {
          this.snackBar.open('No similar images found', 'Close', {
            duration: 3000
          });
        }
      }
    } catch (error) {
      console.error('Error performing reverse search:', error);
      this.snackBar.open('Error performing search', 'Close', {
        duration: 3000
      });
    } finally {
      this.isLoading = false;
    }
  }

  createSlideshow(): void {
    // TODO: Implement slideshow functionality
    this.snackBar.open('Slideshow feature coming soon!', 'Close', {
      duration: 3000
    });
  }

  async downloadResults() {
    if (!this.searchResults?.length) {
      this.snackBar.open('No results to download', 'Close', { duration: 3000 });
      return;
    }
    const zip = new JSZip();
    for (const result of this.searchResults) {
      const file = this.uploadedFiles.find(f => f.name === result.filename.split('/').pop());
      if (file) {
        const fileData = await file.arrayBuffer();
        zip.file(file.name, fileData);
      }
    }
    const content = await zip.generateAsync({ type: 'blob' });
    saveAs(content, 'reverse_search_results.zip');
  }

  ngOnDestroy(): void {
    // Clean up preview URLs
    if (this.previewUrl) {
      URL.revokeObjectURL(this.previewUrl);
    }
    this.previewUrls.forEach(url => URL.revokeObjectURL(url));
    this.previewUrls.clear();
  }
} 
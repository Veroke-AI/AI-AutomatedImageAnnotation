import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
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
  description: string;
  total_images: number;
  processed_images: number;
  results: SearchResult[];
}

@Component({
  selector: 'app-text-search',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule
  ],
  template: `
    <div class="p-6">
      <div class="mb-6">
        <h2 class="text-lg font-medium text-gray-900 mb-2">Text-Based Image Search</h2>
        <p class="text-sm text-gray-600">Find images that match your text description using semantic search</p>
      </div>

      <!-- Search Form -->
      <div class="flex gap-4 mb-6">
        <mat-form-field appearance="fill" class="flex-grow">
          <mat-label>Search Query</mat-label>
          <input 
            matInput 
            [(ngModel)]="searchQuery"
            placeholder="e.g. a boy singing"
          >
        </mat-form-field>

        <button 
          mat-raised-button 
          color="primary"
          [disabled]="!searchQuery.trim() || isLoading || !uploadedFiles?.length"
          (click)="search()"
          style="height: 56px;"
        >
          <span class="flex items-center"><mat-icon>search</mat-icon> Search</span>
        </button>
      </div>

      <!-- Results Stats -->
      <div *ngIf="searchResponse" class="mb-4">
        <p class="text-sm text-gray-600">
          Found {{ searchResponse?.results?.length || 0 }} matches from {{ searchResponse?.total_images || 0 }} images
        </p>
      </div>

      <!-- Results Grid -->
      <div *ngIf="searchResponse?.results?.length" class="space-y-4">
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
          <div *ngFor="let result of searchResponse?.results" class="relative group">
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
      <div *ngIf="hasSearched && (!searchResponse?.results?.length)" class="text-center py-8">
        <mat-icon class="text-gray-400" style="font-size: 48px; width: 48px; height: 48px;">search_off</mat-icon>
        <p class="mt-2 text-gray-600">No images found matching your description</p>
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
export class TextSearchComponent {
  @Input() uploadedFiles: File[] = [];
  searchQuery = '';
  searchResponse: SearchResponse | null = null;
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

  async search(): Promise<void> {
    if (!this.searchQuery.trim() || !this.uploadedFiles?.length) return;

    const formData = new FormData();
    formData.append('description', this.searchQuery);
    
    // Add all uploaded images to formData
    this.uploadedFiles.forEach(file => {
      formData.append('images', file);
    });

    try {
      this.isLoading = true;
      const response = await this.http.post<SearchResponse>(`http://${config.IP}:${config.PORT}/search_images`, formData).toPromise();

      if (response) {
        this.searchResponse = response;
        this.hasSearched = true;

        if (!response.results?.length) {
          this.snackBar.open('No images found matching your description', 'Close', {
            duration: 3000
          });
        }
      }
    } catch (error) {
      console.error('Error performing text search:', error);
      this.snackBar.open('Error performing search', 'Close', {
        duration: 3000
      });
    } finally {
      this.isLoading = false;
    }
  }

  async downloadResults() {
    if (!this.searchResponse?.results?.length) {
      this.snackBar.open('No results to download', 'Close', { duration: 3000 });
      return;
    }
    const zip = new JSZip();
    for (const result of this.searchResponse.results) {
      const file = this.uploadedFiles.find(f => f.name === result.filename.split('/').pop());
      if (file) {
        const fileData = await file.arrayBuffer();
        zip.file(file.name, fileData);
      }
    }
    const content = await zip.generateAsync({ type: 'blob' });
    saveAs(content, 'search_results.zip');
  }

  createSlideshow(): void {
    // TODO: Implement slideshow functionality
    this.snackBar.open('Slideshow feature coming soon!', 'Close', {
      duration: 3000
    });
  }

  ngOnDestroy(): void {
    // Clean up preview URLs
    this.previewUrls.forEach(url => URL.revokeObjectURL(url));
    this.previewUrls.clear();
  }
} 
import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTabsModule } from '@angular/material/tabs';
import { Router } from '@angular/router';

// Import feature components
import { TextSearchComponent } from './text-search/text-search.component';
import { ReverseSearchComponent } from './reverse-search/reverse-search.component';
import { DuplicateDetectionComponent } from './duplicate-detection/duplicate-detection.component';
import { ImageClusteringComponent } from './image-clustering/image-clustering.component';
import { ImageUploaderComponent } from './shared/image-uploader/image-uploader.component';

@Component({
  selector: 'app-visual-intelligence',
  standalone: true,
  imports: [
    CommonModule,
    MatButtonModule,
    MatIconModule,
    MatTabsModule,
    TextSearchComponent,
    ReverseSearchComponent,
    DuplicateDetectionComponent,
    ImageClusteringComponent,
    ImageUploaderComponent
  ],
  template: `
    <div class="min-h-screen bg-gray-100">
      <!-- Header -->
      <header class="bg-white shadow-sm">
        <div class="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8 flex justify-between items-center">
          <h1 class="text-2xl font-semibold text-gray-900" style="margin: 0;">Visual Intelligence</h1>
          <button 
            mat-raised-button 
            color="primary"
            (click)="goBack()"
          >
            <mat-icon>arrow_back</mat-icon>
            Back to Home
          </button>
        </div>
      </header>

      <main class="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        <!-- Image Upload Section -->
        <div class="bg-white rounded-lg shadow p-6 mb-6">
          <h2 class="text-lg font-medium text-gray-900 mb-4">Upload Images</h2>
          <app-image-uploader
            (imagesUploaded)="onImagesUploaded($event)"
          ></app-image-uploader>
        </div>

        <!-- Features Tabs -->
        <div class="bg-white rounded-lg shadow">
          <mat-tab-group>
            <!-- Text Search Tab -->
            <mat-tab label="Text Search">
              <app-text-search [uploadedFiles]="uploadedFiles"></app-text-search>
            </mat-tab>

            <!-- Reverse Search Tab -->
            <mat-tab label="Reverse Search">
              <app-reverse-search [uploadedFiles]="uploadedFiles"></app-reverse-search>
            </mat-tab>

            <!-- Duplicate Detection Tab -->
            <mat-tab label="Duplicate Detection">
              <app-duplicate-detection [uploadedFiles]="uploadedFiles"></app-duplicate-detection>
            </mat-tab>

            <!-- Image Clustering Tab -->
            <mat-tab label="Image Clustering">
              <app-image-clustering [uploadedFiles]="uploadedFiles"></app-image-clustering>
            </mat-tab>
          </mat-tab-group>
        </div>
      </main>

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
export class VisualIntelligenceComponent {
  uploadedFiles: File[] = [];
  isLoading = false;

  constructor(private router: Router) {}

  onImagesUploaded(files: File[]): void {
    this.uploadedFiles = files;
  }

  goBack(): void {
    this.router.navigate(['/']);
  }
} 
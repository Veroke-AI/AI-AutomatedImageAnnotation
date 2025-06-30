import { Component, Input, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatChipsModule } from '@angular/material/chips';
import { MatSnackBar } from '@angular/material/snack-bar';
import { HttpClient } from '@angular/common/http';
import { config } from '../../../../../config';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';

interface ImageCluster {
  clusterId: string;
  count: number;
  label: string;
  sampleImages: string[];
}

@Component({
  selector: 'app-image-clustering',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatChipsModule
  ],
  template: `
    <div class="p-6">
      <div class="mb-6">
        <h2 class="text-lg font-medium text-gray-900 mb-2">Image Clustering</h2>
        <p class="text-sm text-gray-600">Group similar images into clusters based on visual features</p>
      </div>

      <!-- Clustering Controls -->
      <div class="mb-6 space-y-4">
        <div class="flex items-end gap-4">
          <mat-form-field appearance="fill" style="width: 400px;">
            <mat-label>Number of Clusters</mat-label>
            <input 
              matInput 
              type="number"
              [(ngModel)]="numClusters"
              min="2"
              max="20"
            >
            <mat-hint>2-20 clusters</mat-hint>
          </mat-form-field>

          <mat-form-field appearance="fill" class="flex-grow">
            <mat-label>Class Names (comma-separated)</mat-label>
            <input 
              matInput 
              [(ngModel)]="classNames"
              placeholder="e.g. people, cars, buildings"
            >
            <mat-hint>Leave empty to use automatic labels</mat-hint>
          </mat-form-field>

          <button 
            mat-raised-button
            color="primary"
            [disabled]="isLoading || numClusters < 2"
            (click)="clusterImages()"
            style="height: 56px; margin-bottom: 23px;"
          >
            <span class="flex items-center">Generate Clusters</span>
          </button>
        </div>
      </div>

      <!-- Results Table -->
      <div *ngIf="clusters.length > 0" class="space-y-4">
        <h3 class="text-lg font-medium text-gray-900">Clusters ({{ clusters.length }})</h3>
        <div class="bg-white rounded-lg shadow overflow-hidden">
          <table class="min-w-full divide-y divide-gray-200">
            <thead class="bg-gray-50">
              <tr>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Cluster Name</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Image Count</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">File Names</th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Download</th>
              </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
              <tr *ngFor="let cluster of clusters" 
                  class="hover:bg-gray-50 cursor-pointer"
                  (click)="showClusterImages(cluster)">
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{{ cluster.label }}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{{ cluster.count }}</td>
                <td class="px-6 py-4 text-sm text-gray-500 max-w-lg truncate">{{ cluster.sampleImages.join(', ') }}</td>
                <td>
                  <button mat-icon-button (click)="downloadCluster(cluster); $event.stopPropagation()" matTooltip="Download cluster as zip">
                    <mat-icon>download</mat-icon>
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- No Results Message -->
      <div *ngIf="hasSearched && clusters.length === 0" class="text-center py-8">
        <mat-icon class="text-gray-400" style="font-size: 48px; width: 48px; height: 48px;">category</mat-icon>
        <p class="mt-2 text-gray-600">No clusters generated yet</p>
      </div>

      <!-- Image Preview Dialog -->
      <div *ngIf="selectedCluster" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div class="bg-white rounded-lg p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
          <div class="flex justify-between items-center mb-4">
            <h3 class="text-lg font-medium text-gray-900">{{ selectedCluster.label }}</h3>
            <div class="flex items-center gap-2">
              <button mat-icon-button (click)="downloadCluster(selectedCluster)">
                <mat-icon>download</mat-icon>
              </button>
              <button mat-icon-button (click)="selectedCluster = null">
                <mat-icon>close</mat-icon>
              </button>
            </div>
          </div>
          <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div *ngFor="let imagePath of selectedCluster.sampleImages" class="aspect-square relative">
              <img 
                [src]="getImagePreviewUrl(imagePath)"
                class="w-full h-full object-cover rounded-lg"
                [alt]="imagePath"
              >
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
export class ImageClusteringComponent implements OnDestroy {
  @Input() uploadedFiles: File[] = [];
  numClusters = 5;
  classNames = '';
  clusters: ImageCluster[] = [];
  isLoading = false;
  hasSearched = false;
  selectedCluster: ImageCluster | null = null;
  private previewUrls = new Map<string, string>();

  constructor(
    private http: HttpClient,
    private snackBar: MatSnackBar
  ) {
    console.log('ImageClusteringComponent constructed');
  }

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

  async clusterImages(): Promise<void> {
    console.log('clusterImages called');
    if (this.numClusters < 2) {
      this.snackBar.open('Please specify at least 2 clusters', 'Close', { duration: 3000 });
      return;
    }
    if (!this.uploadedFiles?.length) {
      this.snackBar.open('Please upload images first', 'Close', { duration: 3000 });
      return;
    }

    const formData = new FormData();
    formData.append('num_clusters', this.numClusters.toString());
    if (this.classNames.trim()) {
      const classNamesArray = this.classNames.split(',').map(name => name.trim()).filter(name => name);
      formData.append('class_names', JSON.stringify(classNamesArray));
    }
    this.uploadedFiles.forEach(file => {
      formData.append('images', file);
    });

    try {
      this.isLoading = true;
      const response = await this.http.post<any>(`http://${config.IP}:${config.PORT}/cluster_images`, formData).toPromise();
      console.log('API Response:', response);

      // Transform the response
      this.clusters = Object.entries(response.clusters || {}).map(([clusterName, value]: [string, any]) => ({
        clusterId: value.cluster_id.toString(),
        count: value.image_count,
        label: clusterName,
        sampleImages: (value.images || []).map((img: any) => img.filename)
      }));
      console.log('Transformed clusters:', this.clusters);

      this.hasSearched = true;
      if (this.clusters.length === 0) {
        this.snackBar.open('No clusters generated', 'Close', { duration: 3000 });
      }
    } catch (error) {
      console.error('Error clustering images:', error);
      this.snackBar.open('Error generating clusters', 'Close', { duration: 3000 });
    } finally {
      this.isLoading = false;
    }
  }

  showClusterImages(cluster: ImageCluster): void {
    this.selectedCluster = cluster;
  }

  async downloadCluster(cluster: ImageCluster) {
    const zip = new JSZip();
    const folder = zip.folder(cluster.label.replace(/[^a-zA-Z0-9-_ ]/g, '_'))!;

    for (const filename of cluster.sampleImages) {
      const file = this.uploadedFiles.find(f => f.name === filename.split('/').pop());
      if (file) {
        const fileData = await file.arrayBuffer();
        folder.file(file.name, fileData);
      }
    }

    const content = await zip.generateAsync({ type: 'blob' });
    saveAs(content, `${cluster.label}.zip`);
  }

  ngOnDestroy(): void {
    // Clean up preview URLs
    this.previewUrls.forEach(url => URL.revokeObjectURL(url));
    this.previewUrls.clear();
  }
}
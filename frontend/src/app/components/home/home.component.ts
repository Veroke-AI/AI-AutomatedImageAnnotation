import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    MatButtonModule,
    MatIconModule
  ],
  template: `
    <div class="min-h-screen bg-gray-100">
      <header class="bg-white shadow-sm">
        <div class="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <h1 class="text-2xl font-semibold text-gray-900">Image Processing Tools</h1>
        </div>
      </header>

      <main class="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
          <!-- Image Annotation Card -->
          <div class="bg-white rounded-lg shadow overflow-hidden">
            <div class="p-6">
              <div class="flex items-center">
                <div class="flex-shrink-0">
                  <mat-icon class="text-blue-500" style="font-size: 32px; width: 32px; height: 32px;">
                    edit
                  </mat-icon>
                </div>
                <div class="ml-4">
                  <h2 class="text-lg font-medium text-gray-900">Image Annotation</h2>
                  <p class="mt-1 text-sm text-gray-500">
                    Annotate images with bounding boxes and polygons for object detection and segmentation
                  </p>
                </div>
              </div>
              <div class="mt-4">
                <button 
                  mat-raised-button 
                  color="primary"
                  routerLink="/annotate"
                >
                  Start Annotating
                </button>
              </div>
            </div>
          </div>

          <!-- Visual Intelligence Card -->
          <div class="bg-white rounded-lg shadow overflow-hidden">
            <div class="p-6">
              <div class="flex items-center">
                <div class="flex-shrink-0">
                  <mat-icon class="text-blue-500" style="font-size: 32px; width: 32px; height: 32px;">
                    auto_awesome
                  </mat-icon>
                </div>
                <div class="ml-4">
                  <h2 class="text-lg font-medium text-gray-900">Visual Intelligence</h2>
                  <p class="mt-1 text-sm text-gray-500">
                    Search, analyze, and organize images using AI-powered tools
                  </p>
                </div>
              </div>
              <div class="mt-4">
                <button 
                  mat-raised-button 
                  color="primary"
                  routerLink="/visual-intelligence"
                >
                  Explore Tools
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  `,
  styles: [`
    :host {
      display: block;
    }
  `]
})
export class HomeComponent {} 
import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-landing-page',
  template: `
    <div class="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 flex items-center justify-center">
      <div class="max-w-4xl w-full mx-4">
        <h1 class="text-4xl md:text-6xl font-bold text-white text-center mb-12">
          Image Processing Tool
        </h1>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
          <!-- Annotate Images Card -->
          <div 
            class="bg-white rounded-xl shadow-2xl p-8 transform transition-all duration-300 hover:scale-105 hover:shadow-3xl cursor-pointer"
            (click)="navigateTo('annotate')"
          >
            <div class="flex flex-col items-center">
              <div class="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mb-6">
                <mat-icon class="text-blue-600 text-4xl" style="height: 43px; width: 43px;">edit</mat-icon>
              </div>
              <h2 class="text-2xl font-semibold text-gray-800 mb-4">Annotate Images</h2>
              <p class="text-gray-600 text-center">
                Upload and annotate images with click and polygon tools. Perfect for object detection and segmentation tasks.
              </p>
            </div>
          </div>

          <!-- Clean Images Card -->
          <div 
            class="bg-white rounded-xl shadow-2xl p-8 transform transition-all duration-300 hover:scale-105 hover:shadow-3xl cursor-pointer"
            (click)="navigateTo('clean')"
          >
            <div class="flex flex-col items-center">
              <div class="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mb-6">
                <mat-icon class="text-green-600 text-4xl" style="height: 43px; width: 43px;">cleaning_services</mat-icon>
              </div>
              <h2 class="text-2xl font-semibold text-gray-800 mb-4">Clean Images</h2>
              <p class="text-gray-600 text-center">
                Process and clean your images with various preprocessing techniques. Coming soon!
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    :host {
      display: block;
      height: 100vh;
    }
  `]
})
export class LandingPageComponent {
  constructor(private router: Router) {}

  navigateTo(route: string): void {
    if (route === 'annotate') {
      this.router.navigate(['/annotate']);
    } else if (route === 'clean') {
      // Will be implemented later
      console.log('Clean images feature coming soon!');
    }
  }
} 
import { Component, Inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatDialogRef, MAT_DIALOG_DATA, MatDialogModule } from '@angular/material/dialog';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatDividerModule } from '@angular/material/divider';
import { CocoCategory } from '../../services/coco-dataset.service';

@Component({
  selector: 'app-class-selection-dialog',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatDialogModule,
    MatButtonModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatDividerModule
  ],
  template: `
    <h2 mat-dialog-title>Select or Create Class</h2>
    <mat-dialog-content>
      <div class="class-selection-container">
        <!-- Existing Class Selection -->
        <mat-form-field appearance="fill" class="full-width">
          <mat-label>Select Existing Class</mat-label>
          <mat-select [(ngModel)]="selectedCategoryId">
            <mat-option [value]="null">Create New Class</mat-option>
            <mat-option *ngFor="let category of data.categories" [value]="category.id">
              {{ category.name }}
            </mat-option>
          </mat-select>
        </mat-form-field>

        <!-- New Class Creation -->
        <div *ngIf="selectedCategoryId === null" class="new-class-form">
          <mat-divider class="my-3 text-center"></mat-divider>
          <h3 class="text-center">Or</h3>
          <mat-divider class="my-3"></mat-divider>
          
          <mat-form-field appearance="fill" class="full-width">
            <mat-label>Create New Class</mat-label>
            <input matInput [(ngModel)]="newClassName" placeholder="Enter class name">
          </mat-form-field>

          <!-- <mat-form-field appearance="fill" class="full-width">
            <mat-label>Supercategory (Optional)</mat-label>
            <input matInput [(ngModel)]="newClassSupercategory" placeholder="Enter supercategory">
          </mat-form-field> -->
        </div>
      </div>
    </mat-dialog-content>
    <mat-dialog-actions align="end">
      <button mat-button (click)="onCancel()">Cancel</button>
      <button mat-raised-button color="primary" (click)="onConfirm()" [disabled]="!isValid">
        Confirm
      </button>
    </mat-dialog-actions>
  `,
  styles: [`
    .class-selection-container {
      min-width: 300px;
      padding: 8px;
    }
    .full-width {
      width: 100%;
    }
    .my-3 {
      margin: 1rem 0;
    }
    .new-class-form {
      margin-top: 1rem;
    }
  `]
})
export class ClassSelectionDialogComponent {
  selectedCategoryId: number | null = null;
  newClassName = '';
  newClassSupercategory = '';

  constructor(
    public dialogRef: MatDialogRef<ClassSelectionDialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data: { categories: CocoCategory[] }
  ) {}

  get isValid(): boolean {
    return this.selectedCategoryId !== null || (this.newClassName.trim().length > 0);
  }

  onCancel(): void {
    this.dialogRef.close();
  }

  onConfirm(): void {
    if (this.selectedCategoryId !== null) {
      // Return existing category
      this.dialogRef.close({
        type: 'existing',
        categoryId: this.selectedCategoryId
      });
    } else if (this.newClassName.trim()) {
      // Return new category data
      this.dialogRef.close({
        type: 'new',
        name: this.newClassName.trim(),
        supercategory: this.newClassSupercategory.trim() || undefined
      });
    }
  }
} 
import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { LandingPageComponent } from './components/landing-page/landing-page.component';
import { AnnotationWorkspaceComponent } from './components/annotation-workspace/annotation-workspace.component';

const routes: Routes = [
  { path: '', component: LandingPageComponent },
  { path: 'annotate', component: AnnotationWorkspaceComponent },
  { path: '**', redirectTo: '' }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { } 
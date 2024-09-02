# Reference material: https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-28-practical-post-process-depth-field
#%%
class DOFCalculator:
    def __init__(self, focal_len_mm: float, aperature_radius_mm: float):
        self.focal_len_mm = focal_len_mm
        self.aperature_radius_mm = aperature_radius_mm
    
    @staticmethod
    def from_fstop(focal_len_mm: float, f_stop: float):
        aperature_radius_mm = focal_len_mm/(2*f_stop)
        return DOFCalculator(focal_len_mm, aperature_radius_mm)
        
    def circle_of_confusion(self, distance_meters: float, focus_distance_mm: float):
        '''
        distance_meters: distance from the camera to the object in meters
        focus_distance_mm: distance from the lens to the image sensor in mm (this is what changes when the camera changes focus)

        Equations taken from https://en.wikipedia.org/wiki/Circle_of_confusion
        '''
        distance_mm = distance_meters*1000

        f = self.focal_len_mm
        S1 = (f * focus_distance_mm) / (focus_distance_mm - f)
        S2 = distance_mm
        A = 2 * self.aperature_radius_mm
        return A * np.abs(S2-S1)/S2 * f/(S1-f)



if __name__ == "__main__":
    print("DOF Calculator")
    # this comment makes the code a cell you can run in an integrated vscode notebook
    #%%
    from matplotlib import pyplot as plt
    import numpy as np
    calculator = DOFCalculator.from_fstop(714, 7)
    # make 2d plot of circle of confusion as function of object distance and focus distance
    plt.figure()
    plt.title(f'Focal length: {calculator.focal_len_mm}mm, aperature radius: {calculator.aperature_radius_mm} mm')
    GRID_RESOLUTION = 400
    def x_scaling_fn(x):
        return 1/x
    def x_scaling_fn_inv(x):
        return 1/x
    xs, ys = np.meshgrid(
        np.linspace(x_scaling_fn(10_000), x_scaling_fn(10), GRID_RESOLUTION), # object distance
        np.linspace(714,714+60, GRID_RESOLUTION) # focus plane location
    )
    plt.xlabel('Object distance (m)')
    plt.ylabel('Focus distance (mm)')
    def circle_radius_to_num_pixels(radius):
        pixel_area = 5.6 * 3.1 / (1920*1080) # mm^2
        return np.pi * radius**2 / pixel_area
    radii = np.array([
            np.clip(circle_radius_to_num_pixels(
                calculator.circle_of_confusion(x_scaling_fn_inv(x), y)
            ), 0, 100)
            for x,y in zip(xs.ravel(), ys.ravel())
    ])
    print(np.min(radii), np.max(radii))
    plt.imshow(radii.reshape((GRID_RESOLUTION,GRID_RESOLUTION)))
    plt.xticks(np.linspace(0,GRID_RESOLUTION, 5), x_scaling_fn_inv(np.linspace(xs.min(),xs.max(), 5)).astype(int))
    plt.yticks(np.linspace(0,GRID_RESOLUTION, 5), np.linspace(ys.min(),ys.max(), 5))
    plt.colorbar()
    plt.show()

    
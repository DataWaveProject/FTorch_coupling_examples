program inference

   ! Import precision info from iso
   use, intrinsic :: iso_fortran_env, only : sp => real32

   ! Import our library for interfacing with PyTorch
   use ftorch, only : torch_model, torch_tensor, torch_kCPU, torch_delete, &
                      torch_tensor_from_array, torch_model_load, torch_model_forward

   ! Import our tools module for testing utils
   use ftorch_test_utils, only : assert_allclose

   implicit none

   integer, parameter :: wp = sp

   integer, parameter :: B = 50, C = 1, H = 28, W = 28
   integer, parameter :: in_dims = 4
   integer, parameter :: in_shape(in_dims) = [B, C, H, W]
   integer, parameter :: out_dims = 4
   integer, parameter :: out_shape(out_dims) = [B, C, H, W]
   ! real(wp) :: x_real, true_sine, prediction, error

   integer :: num_args, ix, i
   character(len=128), dimension(:), allocatable :: args

   ! Set up Fortran data structures
   real(wp), allocatable, dimension(:,:,:,:) :: in_data
   real(wp), allocatable, dimension(:,:,:,:) :: ref_data
   real(wp), allocatable, dimension(:,:,:,:) :: out_data
   real(wp), allocatable :: x_vals(:), true_sine(:), error(:)
   real(wp), parameter :: tol = 0.1
   real(wp), dimension(B) :: per_sample_mae
   real(wp) :: total_mae

   ! Set up Torch data structures
   ! The net, a vector of input tensors (in this case we only have one), and the output tensor
   type(torch_model) :: model
   type(torch_tensor), dimension(1) :: in_tensors
   type(torch_tensor), dimension(1) :: out_tensors

   ! Flag for testing
   logical :: test_pass

   ! Get TorchScript model file as a command line argument
   num_args = command_argument_count()
   allocate(args(num_args))
   do ix = 1, num_args
       call get_command_argument(ix,args(ix))
   end do

   allocate(in_data(in_shape(1), in_shape(2), in_shape(3), in_shape(4)))
   allocate(ref_data(in_shape(1), in_shape(2), in_shape(3), in_shape(4)))
   allocate(out_data(out_shape(1), out_shape(2), out_shape(3), out_shape(4)))

   ! Create random data
   call random_seed()
   call random_number(in_data)

   in_data = 2.0_wp * in_data - 1.0_wp  ! Scale from [0,1] → [-1,1]
   ref_data = in_data + 1.0_wp

   ! Create Torch input/output tensors from the above arrays
   call torch_tensor_from_array(in_tensors(1), in_data, torch_kCPU)
   call torch_tensor_from_array(out_tensors(1), out_data, torch_kCPU)

   ! Load ML model
   call torch_model_load(model, args(1), torch_kCPU)

   ! Check shape of in_tensors
   print *, "Input array shape: ", shape(in_data)

   ! Infer
   call torch_model_forward(model, in_tensors, out_tensors)

   do i = 1, B
     per_sample_mae(i) = sum(abs(out_data(i,1,:,:) - ref_data(i,1,:,:))) / real(H*W, wp)
   end do

   ! Average over all samples
   total_mae = sum(per_sample_mae) / real(B, wp)

   if (total_mae < tol) then
     print *, "PASS: MAE is within tolerance:", total_mae
   else
     print *, "FAIL: MAE exceeds tolerance:", total_mae
   end if

   ! Cleanup
   call torch_delete(model)
   call torch_delete(in_tensors)
   call torch_delete(out_tensors)

end program inference

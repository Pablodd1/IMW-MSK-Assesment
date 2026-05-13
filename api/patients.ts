export default function handler(req: any, res: any) {
  const patients = [
    { id: 1, first_name: 'John', last_name: 'Smith', date_of_birth: '1979-05-15', gender: 'male', email: 'john.smith@email.com', phone: '(305) 555-0123', created_at: '2025-01-15T10:30:00Z' },
    { id: 2, first_name: 'Maria', last_name: 'Garcia', date_of_birth: '1963-08-22', gender: 'female', email: 'maria.garcia@email.com', phone: '(305) 555-0234', created_at: '2025-02-20T11:00:00Z' },
    { id: 3, first_name: 'David', last_name: 'Chen', date_of_birth: '1987-03-10', gender: 'male', email: 'david.chen@email.com', phone: '(305) 555-0345', created_at: '2025-03-05T09:30:00Z' }
  ]
  res.status(200).json({ success: true, data: patients })
}
